#!/usr/bin/env python3
"""
BUILD-TIME script: distils the Hugging Face Indian-law corpus into a compact,
offline, deterministic classifier that the AEGIS server can run with sklearn only.

Sources (all openly licensed, downloaded into legal_kb/):
  * Hanno-Labs/indian-ipc-statute-identification  (CC-BY-4.0)
      40,001 fact-narrative -> IPC-section pairs from Indian High Court judgments.
      Section references inside the facts are masked as [STATUTE] by the publisher,
      so this is a genuine facts->statute task and cannot be string-matched.
  * GSMS-B/indian-legal-sections-bns-bnss-bsa-2023 (Apache-2.0)
      Full operative text of BNS / BNSS / BSA 2023 sections.
  * GSMS-B/ipc-bns-crpc-bnss-iea-bsa-...-mapping (CC-BY-4.0)
      Authoritative old-act -> new-act section correspondence.

Runtime needs ONLY joblib + scikit-learn (both pre-installed and persistent).
pyarrow/pandas are required here at build time only.
"""
import argparse
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_kb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ---------------------------------------------------------------- fact cleaning
BOILER_ANCHORS = [
    r"heard\s+learned\s+counsel",
    r"oral\s+order",
    r"coram\s*:",
    r"appearance\s*:",
    r"per\s*:\s*honourable",
    r"order\s+dated",
]
NOISE_PATTERNS = [
    (r"=+", " "),
    (r"-{4,}", " "),
    (r"\.{3,}", " "),
    (r"\[STATUTE\]", " statute "),
    (r"IN THE HIGH COURT OF JUDICATURE AT [A-Z ]+", " "),
    (r"(Crl\.|Criminal) (Miscellaneous|Misc\.) No\.?\s*\d+\s*(of|/)\s*\d{4}", " "),
    (r"C\.?W\.?\s*No\.?\s*\d+", " "),
    (r"Criminal Appeal No\.?\s*\d+", " "),
    (r"Hon'?ble\s+(Mr\.?)?\s*Justice\s+[A-Z][A-Za-z. ]{2,40}", " "),
    (r"HONOURABLE (MR\.|MRS\.|MS\.)? ?JUSTICE [A-Z][A-Z. ]{2,40}", " "),
    (r"Mr\.?\s+[A-Z][A-Za-z.]{2,30}(,\s*(Advocate|Sr\.?\s*Advocate|APP|A\.?G\.?A\.?))?", " "),
    (r"(For|On behalf of) the (Petitioner|Respondent|Opposite Party|Applicant|State)[/a-z ]*", " "),
    (r"\b(Petitioner|Respondent|Opposite Party|Applicant|Appellant)[/s]*\b", " "),
    (r"\bR/o village[- ][^,.]{2,40}", " "),
    (r"\bP\.?S\.?-?\s*[A-Z][A-Za-z ]{2,30}", " "),
    (r"\bDistrict[- ][A-Z][A-Za-z]{2,25}", " "),
    (r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " "),
    (r"\b\d{4,}\b", " "),
    (r"\s{2,}", " "),
]


def clean_facts(t: str) -> str:
    """Strip judgment boilerplate, keep the factual narrative."""
    if not isinstance(t, str):
        return ""
    low = t.lower()
    cut = 0
    for a in BOILER_ANCHORS:
        m = re.search(a, low)
        if m and m.end() > cut:
            cut = m.end()
    body = t[cut:] if cut else t
    for pat, rep in NOISE_PATTERNS:
        body = re.sub(pat, rep, body, flags=re.I)
    body = re.sub(r"\s+", " ", body).strip(" .,;:-")
    return body[:3000]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="subsample rows (0 = all)")
    ap.add_argument("--min-samples", type=int, default=40, help="drop rarer sections")
    ap.add_argument("--quick", action="store_true", help="skip probability calibration")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    print("[1/6] loading HF corpus …")
    df = pd.read_parquet(os.path.join(KB, "ipc_train.parquet"))
    print(f"      {len(df)} rows, {df['ipc_section'].nunique()} distinct IPC sections")

    print("[2/6] cleaning judgment boilerplate …")
    df["facts"] = df["case_facts"].map(clean_facts)
    df = df[df["facts"].str.len() > 120].copy()
    print(f"      {len(df)} rows retained after cleaning")

    vc = df["ipc_section"].value_counts()
    keep = vc[vc >= args.min_samples].index.tolist()
    df = df[df["ipc_section"].isin(keep)].copy()
    print(f"      kept {len(keep)} sections with >= {args.min_samples} samples "
          f"({len(df)} rows, {len(df)/40001*100:.1f}% of corpus)")

    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)
        print(f"      subsampled to {len(df)} rows")

    X = df["facts"].tolist()
    y = df["ipc_section"].astype(str).tolist()
    # 3-way split: train / calibration (disjoint, for prefit sigmoid) / held-out test
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.60, random_state=42, stratify=yrest)
    print(f"      train {len(Xtr)} / calib {len(Xcal)} / held-out test {len(Xte)}")

    print("[3/6] vectorising (word 1-2 grams + char_wb 3-5 grams) …")
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.92,
        sublinear_tf=True, max_features=90000, strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_df=0.92,
        sublinear_tf=True, max_features=70000,
    )
    from scipy.sparse import hstack

    vec.fit(Xtr); char_vec.fit(Xtr)

    def featurise(rows):
        return hstack([vec.transform(rows), char_vec.transform(rows)]).tocsr()

    Xtr_f = featurise(Xtr)
    print(f"      feature matrix {Xtr_f.shape}")

    print("[4/6] training LinearSVC …")
    svc = LinearSVC(C=0.45, class_weight="balanced", max_iter=6000, dual=True)
    svc.fit(Xtr_f, ytr)
    del Xtr_f
    import gc; gc.collect()          # free the largest matrix before building the others
    Xcal_f = featurise(Xcal)
    Xte_f = featurise(Xte)
    top1 = accuracy_score(yte, svc.predict(Xte_f))
    print(f"      held-out top-1 accuracy = {top1*100:.2f}%")

    # top-5 accuracy from decision scores
    dec = svc.decision_function(Xte_f)
    top5_idx = np.argsort(-dec, axis=1)[:, :5]
    classes = np.array(svc.classes_)
    hit = 0
    for i, gold in enumerate(yte):
        if gold in classes[top5_idx[i]]:
            hit += 1
    top5 = hit / len(yte)
    print(f"      held-out top-5 accuracy = {top5*100:.2f}%")

    if args.quick:
        clf = svc
        print("      (quick mode: no probability calibration — scores are decision values)")
    else:
        print("[5/6] calibrating probabilities (sigmoid, cv=prefit on disjoint calib set) …")
        clf = CalibratedClassifierCV(svc, method="sigmoid", cv="prefit")
        clf.fit(Xcal_f, ycal)
        proba = clf.predict_proba(Xte_f)
        ctop1 = accuracy_score(yte, clf.classes_[proba.argmax(1)])
        print(f"      calibrated top-1 = {ctop1*100:.2f}%")

    print("[6/6] persisting bundle …")
    mapping = json.load(open(os.path.join(KB, "ipc_to_bns.json")))
    bns = json.load(open(os.path.join(KB, "bns_sections.json")))
    bsa = json.load(open(os.path.join(KB, "bsa_sections.json")))
    iea_bsa = json.load(open(os.path.join(KB, "iea_to_bsa.json")))

    bundle = {
        "word_vec": vec,
        "char_vec": char_vec,
        "clf": clf,
        "classes": [str(c) for c in (clf.classes_ if hasattr(clf, "classes_") else svc.classes_)],
        "ipc_to_bns": mapping,
        "bns_sections": bns,
        "bsa_sections": bsa,
        "iea_to_bsa": iea_bsa,
        "metrics": {
            "train_rows": int(len(Xtr)), "calib_rows": int(len(Xcal)), "test_rows": int(len(Xte)),
            "sections": int(len(keep)),
            "top1": round(float(top1) * 100, 2), "top5": round(float(top5) * 100, 2),
            "calibrated": not args.quick,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_datasets": [
                "Hanno-Labs/indian-ipc-statute-identification (CC-BY-4.0)",
                "GSMS-B/indian-legal-sections-bns-bnss-bsa-2023 (Apache-2.0)",
                "GSMS-B/ipc-bns-crpc-bnss-iea-bsa-respective-old-new-sections-mapping (CC-BY-4.0)",
            ],
        },
    }
    path = os.path.join(OUT, "legal_ai.joblib")
    joblib.dump(bundle, path, compress=3)
    print(f"      wrote {path} ({os.path.getsize(path)/1e6:.1f} MB) in {time.time()-t0:.0f}s")
    json.dump(bundle["metrics"], open(os.path.join(OUT, "legal_ai_metrics.json"), "w"), indent=2)
    print("      metrics ->", json.dumps(bundle["metrics"][:4] if False else {
        k: bundle["metrics"][k] for k in ("sections", "top1", "top5", "calibrated")}))


if __name__ == "__main__":
    main()
