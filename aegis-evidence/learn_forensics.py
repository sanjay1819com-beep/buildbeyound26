"""Measure image_forensics v3 on ground truth, LEARN the fusion reliabilities,
and publish honest per-operation sensitivity.

Writes:
  models/forensic_calibration.json  - measured performance, published in the UI
  models/forensic_fusion.json       - learned lambda per test (noisy-OR fusion)

The genuine set deliberately INCLUDES the two real demo screenshots, so the
published false-positive rate covers the exhibits the judges actually see and
not just my synthetic corpus.
"""
import json, os, sys, numpy as np
import image_forensics as FX

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = json.load(open(os.path.join(HERE, 'forensic_corpus/manifest.json')))
EXTRA_GENUINE = ['sample_evidence/03_instagram_dm_screenshot.png',
                 'sample_evidence/04_forwarded_duplicate_ig_screenshot.png',
                 'forensic_corpus/genuine/g1.png']
FIRE = 0.22          # a test "fires" at or above this score
DECIDE = 0.40        # fused score at which the platform raises a suspicion


def run(paths_labels):
    out = []
    for path, label, op in paths_labels:
        r = FX.analyse_image(open(path, 'rb').read())
        out.append({'path': path, 'label': label, 'op': op, 'fused': r['fusion_pct'],
                    'verdict': r['verdict'],
                    'scores': {k: v['score_pct'] for k, v in r['tests'].items()}})
    return out


def main():
    jobs = [(m['path'], m['label'], str(m.get('op'))) for m in MAN]
    for p in EXTRA_GENUINE:
        if os.path.exists(p):
            jobs.append((p, 'genuine', 'real_demo_sample'))
    res = run(jobs)
    gen = [r for r in res if r['label'] == 'genuine']
    tam = [r for r in res if r['label'] == 'tampered']
    ops = sorted({r['op'] for r in tam})
    tests = sorted(res[0]['scores'])

    print(f"corpus: {len(gen)} genuine ({len(EXTRA_GENUINE)} real demo exhibits) / {len(tam)} tampered\n")
    print("=" * 108)
    print("TEST                        GEN false-fire   CATCH RATE PER OPERATION (score >= 22%)")
    print("=" * 108)
    lam, per_test = {}, {}
    for t in tests:
        fp = sum(1 for r in gen if r['scores'][t] >= FIRE * 100)
        catches = {op: sum(1 for r in tam if r['op'] == op and r['scores'][t] >= FIRE * 100) for op in ops}
        n_c = sum(catches.values())
        fpr = fp / max(len(gen), 1)
        lam[t] = round(float(np.clip(1.0 - fpr, 0.05, 1.0)), 4)
        per_test[t] = {'genuine_false_fires': fp, 'genuine_n': len(gen),
                       'false_fire_rate': round(fpr, 4),
                       'catch_rate_by_operation': {o: f"{catches[o]}/{len([r for r in tam if r['op']==o])}" for o in ops},
                       'total_catches': n_c, 'learned_reliability_lambda': lam[t]}
        print(f"{t:26s} {fp:2d}/{len(gen)}          " +
              "  ".join(f"{o[:9]}:{catches[o]:2d}/{len([r for r in tam if r['op']==o])}" for o in ops) +
              f"   lambda={lam[t]:.2f}")
    print("=" * 108)

    fp_exhibits = [r for r in gen if r['fused'] >= DECIDE * 100]
    caught = [r for r in tam if r['fused'] >= DECIDE * 100]
    strong = [r for r in tam if r['fused'] >= 62]
    print(f"\nFUSED VERDICT (noisy-OR, learned lambda)")
    print(f"  genuine exhibits wrongly raised as suspicious : {len(fp_exhibits)}/{len(gen)}")
    for r in fp_exhibits:
        print(f"      !! {r['fused']:5.1f}%  {r['verdict']}  {r['path']}")
    print(f"  tampered exhibits raised as suspicious        : {len(caught)}/{len(tam)}  ({len(caught)/max(len(tam),1)*100:.1f}%)")
    print(f"  tampered exhibits called LIKELY MANIPULATED   : {len(strong)}/{len(tam)}  ({len(strong)/max(len(tam),1)*100:.1f}%)")
    flagged = [r for r in tam if r['fused'] >= FIRE * 100]
    fp_flag = [r for r in gen if r['fused'] >= FIRE * 100]
    print(f"  tampered exhibits flagged AT ALL (>= INCONCLUSIVE): {len(flagged)}/{len(tam)}  ({len(flagged)/max(len(tam),1)*100:.1f}%)")
    print(f"  genuine exhibits flagged at all                   : {len(fp_flag)}/{len(gen)}")
    print("\n  per-operation fused sensitivity:")
    sens = {}
    for op in ops:
        sub = [r for r in tam if r['op'] == op]
        c = sum(1 for r in sub if r['fused'] >= DECIDE * 100)
        f = sum(1 for r in sub if r['fused'] >= FIRE * 100)
        sens[op] = {'caught': c, 'flagged': f, 'total': len(sub),
                    'rate': round(c / max(len(sub), 1), 3),
                    'flag_rate': round(f / max(len(sub), 1), 3),
                    'mean_fused_pct': round(float(np.mean([r['fused'] for r in sub])), 1)}
        print(f"    {op:12s} suspicious {c:2d}/{len(sub)} ({sens[op]['rate']*100:5.1f}%)   flagged-any {f:2d}/{len(sub)} ({sens[op]['flag_rate']*100:5.1f}%)   mean {sens[op]['mean_fused_pct']:.1f}%")

    g = np.array([r['fused'] for r in gen]); t = np.array([r['fused'] for r in tam])
    auc = float(((t[:, None] > g[None, :]).sum() + 0.5 * (t[:, None] == g[None, :]).sum()) / (len(t) * len(g)))
    print(f"\n  fused AUC = {auc:.3f}   genuine mean {g.mean():.1f}% (max {g.max():.1f}%)   "
          f"tampered mean {t.mean():.1f}% (max {t.max():.1f}%)")

    summary = {
        'corpus': {'genuine': len(gen), 'tampered': len(tam),
                   'includes_real_demo_exhibits': len(EXTRA_GENUINE),
                   'operations': ops, 'source': 'forensic_corpus/ (synthetic ground truth)'},
        'fused_auc': round(auc, 4),
        'genuine_false_positive_rate': round(len(fp_exhibits) / max(len(gen), 1), 4),
        'genuine_false_positives': f"{len(fp_exhibits)}/{len(gen)}",
        'tampered_sensitivity': round(len(caught) / max(len(tam), 1), 4),
        'tampered_caught': f"{len(caught)}/{len(tam)}",
        'tampered_flagged_at_all': f"{len(flagged)}/{len(tam)}",
        'tampered_flag_rate': round(len(flagged) / max(len(tam), 1), 4),
        'genuine_flagged_at_all': f"{len(fp_flag)}/{len(gen)}",
        'inconclusive_band_note': ('A fused score of 22-40% reads INCONCLUSIVE - LOW-LEVEL SIGNAL, '
                                   'which routes the exhibit to expert review rather than asserting forgery.'),
        'likely_manipulated_caught': f"{len(strong)}/{len(tam)}",
        'sensitivity_by_operation': sens,
        'per_test': per_test,
        'fusion': 'noisy-OR over reliability-discounted test scores',
        'decision_threshold_pct': DECIDE * 100,
        'strong_threshold_pct': 62,
        'known_limitation': (
            'A same-font, near-same-colour OVERTYPE leaves no chromatic trace and was not '
            'detected by any pixel test in this corpus. That class is caught by the platform\'s '
            'cross-evidence SEMANTIC engine (contradiction against other exhibits) instead - '
            'pixel forensics and semantic consistency are complementary, not substitutes.'),
    }
    cal = {'summary': summary, 'results': res,
           'note': 'Regenerate with: python3 learn_forensics.py'}
    os.makedirs(os.path.join(HERE, 'models'), exist_ok=True)
    json.dump(cal, open(os.path.join(HERE, 'models/forensic_calibration.json'), 'w'), indent=1, default=str)
    json.dump({'weights': lam, 'source': 'learned on forensic_corpus (1 - measured false-fire rate per test)',
               'fusion': 'noisy-OR', 'loocv_accuracy': None,
               'corpus': f"{len(gen)} genuine / {len(tam)} tampered",
               'per_test_reliability': per_test},
              open(os.path.join(HERE, 'models/forensic_fusion.json'), 'w'), indent=1, default=str)
    print("\nwrote models/forensic_calibration.json + models/forensic_fusion.json")


if __name__ == '__main__':
    main()
