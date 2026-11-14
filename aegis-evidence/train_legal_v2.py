#!/usr/bin/env python3
"""Memory-safe build of the AEGIS Legal AI precedent classifier (2 GB box)."""
import gc
import json
import os
import resource
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import train_legal_ai as T   # reuse clean_facts()

KB = "legal_kb"
OUT = "models"
ROWS = 22000
MIN_SAMPLES = 40


def mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


t0 = time.time()
print(f"[1/7] loading HF corpus …  ({mb():.0f} MB)")
df = pd.read_parquet(os.path.join(KB, "ipc_train.parquet"))
df["facts"] = df["case_facts"].map(T.clean_facts)
df = df[df["facts"].str.len() > 120].copy()
vc = df["ipc_section"].astype(str).value_counts()
keep = vc[vc >= MIN_SAMPLES].index.tolist()
n_sections = len(keep)
df = df[df["ipc_section"].astype(str).isin(keep)].copy()
df["ipc_section"] = df["ipc_section"].astype(str)
print(f"      {len(df)} rows over {len(keep)} IPC sections")

# stratified downsample to the memory budget
if len(df) > ROWS:
    frac = ROWS / len(df)
    df = df.groupby("ipc_section", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(60, int(round(len(g) * frac)))), random_state=42))
print(f"[2/7] stratified subsample -> {len(df)} rows  ({mb():.0f} MB)")

X = df["facts"].tolist()
y = df["ipc_section"].tolist()
del df, vc, keep
gc.collect()          # free the corpus before vectorising
Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.55, random_state=42, stratify=yrest)
print(f"      train {len(Xtr)} / calib {len(Xcal)} / test {len(Xte)}")

print(f"[3/7] vectorising …  ({mb():.0f} MB)")
wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.92,
                     sublinear_tf=True, max_features=70000, dtype=np.float32,
                     strip_accents="unicode")
wv.fit(Xtr)
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=10, max_df=0.92,
                     sublinear_tf=True, max_features=22000, dtype=np.float32)
cv.fit(Xtr)


def feat(rows):
    return hstack([wv.transform(rows), cv.transform(rows)]).tocsr()


Xtr_f = feat(Xtr)
print(f"      train matrix {Xtr_f.shape} nnz={Xtr_f.nnz}  ({mb():.0f} MB)")

print(f"[4/7] training LinearSVC …  ({mb():.0f} MB)")
svc = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
svc.fit(Xtr_f, ytr)
del Xtr_f
gc.collect()
print(f"      fitted, {len(svc.classes_)} classes  ({mb():.0f} MB)")

n_cal = len(Xcal)
Xcal_f = feat(Xcal); Xte_f = feat(Xte)
del Xcal
gc.collect()

top1 = accuracy_score(yte, svc.predict(Xte_f))
dec = svc.decision_function(Xte_f)
classes = np.array(svc.classes_)
top5_idx = np.argsort(-dec, axis=1)[:, :5]
top5 = float(np.mean([yte[i] in classes[top5_idx[i]] for i in range(len(yte))]))
top10_idx = np.argsort(-dec, axis=1)[:, :10]
top10 = float(np.mean([yte[i] in classes[top10_idx[i]] for i in range(len(yte))]))
print(f"[5/7] held-out: top-1 {top1*100:.2f}%  top-5 {top5*100:.2f}%  top-10 {top10*100:.2f}%   ({mb():.0f} MB)")

print(f"[6/7] calibrating (sigmoid, prefit on disjoint set) …  ({mb():.0f} MB)")
clf = CalibratedClassifierCV(svc, method="sigmoid", cv="prefit")
clf.fit(Xcal_f, ycal)
proba = clf.predict_proba(Xte_f)
ctop1 = accuracy_score(yte, clf.classes_[proba.argmax(1)])
print(f"      calibrated top-1 {ctop1*100:.2f}%   ({mb():.0f} MB)")

print("[7/7] persisting …")
os.makedirs(OUT, exist_ok=True)
bundle = {
    "word_vec": wv, "char_vec": cv, "clf": clf,
    "classes": [str(c) for c in clf.classes_],
    "ipc_to_bns": json.load(open(os.path.join(KB, "ipc_to_bns.json"))),
    "bns_sections": json.load(open(os.path.join(KB, "bns_sections.json"))),
    "bsa_sections": json.load(open(os.path.join(KB, "bsa_sections.json"))),
    "iea_to_bsa": json.load(open(os.path.join(KB, "iea_to_bsa.json"))),
    "metrics": {
        "engine": "precedent-classifier",
        "train_rows": int(len(Xtr)), "calib_rows": int(n_cal), "test_rows": int(len(Xte)),
        "sections": int(n_sections), "corpus_rows_available": 40001,
        "top1": round(float(top1) * 100, 2), "top5": round(float(top5) * 100, 2),
        "top10": round(float(top10) * 100, 2), "calibrated_top1": round(float(ctop1) * 100, 2),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "peak_rss_mb": round(mb()),
        "source_datasets": [
            "Hanno-Labs/indian-ipc-statute-identification (CC-BY-4.0) — 40,001 Indian High Court fact narratives -> IPC section, statute refs masked by publisher",
            "GSMS-B/indian-legal-sections-bns-bnss-bsa-2023 (Apache-2.0) — operative text of BNS/BNSS/BSA 2023",
            "GSMS-B/ipc-bns-crpc-bnss-iea-bsa-respective-old-new-sections-mapping (CC-BY-4.0)",
        ],
    },
}
p = os.path.join(OUT, "legal_ai.joblib")
joblib.dump(bundle, p, compress=3)
json.dump(bundle["metrics"], open(os.path.join(OUT, "legal_ai_metrics.json"), "w"), indent=2)
print(f"      wrote {p} ({os.path.getsize(p)/1e6:.1f} MB) in {time.time()-t0:.0f}s, peak {mb():.0f} MB")
print("      ", json.dumps({k: bundle["metrics"][k] for k in ("sections", "top1", "top5", "top10", "calibrated_top1")}))
