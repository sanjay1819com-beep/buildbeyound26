#!/usr/bin/env python3
"""Calibration harness: measures how well each forensic test separates
GENUINE screenshots from KNOWN-TAMPERED ones, and reports honest metrics."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_forensics as IF   # noqa: E402

CORP = "forensic_corpus"


def auc(genuine, tampered):
    """Probability a random tampered image scores above a random genuine one."""
    if not genuine or not tampered:
        return 0.5
    g, t = np.array(genuine, float), np.array(tampered, float)
    wins = sum((ti > gj) + 0.5 * (ti == gj) for ti in t for gj in g)
    return wins / (len(t) * len(g))


def main():
    man = json.load(open(os.path.join(CORP, "manifest.json")))
    per_test = {}
    fused = {"genuine": [], "tampered": []}
    rows = []
    for m in man:
        raw = open(m["path"], "rb").read()
        r = IF.analyse_image(raw)
        rows.append((m["label"], m["op"], r["fusion_pct"], r["verdict"]))
        fused[m["label"]].append(r["fusion_pct"])
        for k, v in r["tests"].items():
            per_test.setdefault(k, {"genuine": [], "tampered": []})[m["label"]].append(v["score_pct"])

    print("=" * 104)
    print(f"{'TEST':34} {'GENUINE mean/max':>18} {'TAMPERED mean/max':>19} {'AUC':>7}  VERDICT")
    print("=" * 104)
    for k, v in per_test.items():
        a = auc(v["genuine"], v["tampered"])
        gm, gx = np.mean(v["genuine"]), np.max(v["genuine"])
        tm, tx = np.mean(v["tampered"]), np.max(v["tampered"])
        flag = "SEPARATES" if a >= 0.80 else ("weak" if a >= 0.62 else "USELESS/INVERTED")
        print(f"{k:34} {gm:8.1f} /{gx:6.1f} {tm:10.1f} /{tx:6.1f} {a:7.3f}  {flag}")
    print("=" * 104)
    a = auc(fused["genuine"], fused["tampered"])
    print(f"{'FUSED (current weights)':34} {np.mean(fused['genuine']):8.1f} /{np.max(fused['genuine']):6.1f}"
          f" {np.mean(fused['tampered']):10.1f} /{np.max(fused['tampered']):6.1f} {a:7.3f}")
    print()
    fp = [(l, o, f, v) for l, o, f, v in rows if l == "genuine" and f >= 45]
    fn = [(l, o, f, v) for l, o, f, v in rows if l == "tampered" and f < 45]
    print(f"FALSE POSITIVES (genuine scored >=45%): {len(fp)}/{sum(1 for r in rows if r[0]=='genuine')}")
    for l, o, f, v in fp:
        print(f"   {f:5.1f}%  {v}")
    print(f"FALSE NEGATIVES (tampered scored <45%): {len(fn)}/{sum(1 for r in rows if r[0]=='tampered')}")
    for l, o, f, v in fn:
        print(f"   {f:5.1f}%  op={o}")
    print()
    print("per-operation detection (tampered fusion %):")
    for op in sorted({o for _, o, _, _ in rows if o}):
        vals = [f for l, o, f, v in rows if o == op]
        print(f"   {op:12} mean {np.mean(vals):5.1f}%  min {min(vals):5.1f}%  max {max(vals):5.1f}%")
    json.dump({"per_test_auc": {k: auc(v["genuine"], v["tampered"]) for k, v in per_test.items()},
               "fused_auc": a,
               "genuine_scores": fused["genuine"], "tampered_scores": fused["tampered"]},
              open("models/forensic_calibration.json", "w"), indent=1)


if __name__ == "__main__":
    main()
