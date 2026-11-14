"""Probe v4 - decision metric is now GENUINE FALSE-POSITIVE RATE + PER-OP CATCH RATE.

Global AUC was the wrong yardstick: each test targets a different tampering
operation, so a clone-specific test with perfect specificity looks "useless"
when averaged over eraser/paste/splice cases. What a court needs is a test that
never fires on a genuine exhibit and reliably fires on the operation it claims.

Fixed here:
  F2  np.diff makes gx one column narrower, so the guard `gx.shape[1] < wb*16`
      returned 0.0 for EVERY image. Block grid is now derived from gx itself.
  F14 exact-colour runs failed because the paste is RESIZED, so its foreign
      colours are interpolation intermediates each below the count floor.
      Reformulated at block level: fraction of pixels outside the document's own
      design palette, then the largest contiguous rectangle of high-fraction blocks.
  F13 tightened to a within-row group of glyphs whose colour is 14-45 away from
      the modal glyph colour and is not itself a design colour.
"""
import json, cv2, numpy as np
from PIL import Image

MAN = json.load(open('forensic_corpus/manifest.json'))
B = 16


def load(p):
    img = Image.open(p)
    return np.array(img.convert('RGB')), img.format


def ink_mask(gray):
    q = (gray // 16).astype(np.uint8)
    bg = int(np.bincount(q.ravel(), minlength=16).argmax()) * 16 + 8
    return np.abs(gray - bg) > 48


def ckeys(rgb):
    return ((rgb[:, :, 0].astype(np.int64) << 20) |
            (rgb[:, :, 1].astype(np.int64) << 10) | rgb[:, :, 2].astype(np.int64))


def dec(k):
    return np.array([(int(k) >> 20) & 1023, (int(k) >> 10) & 1023, int(k) & 1023], float)


# ------------------------------------------------------------------ F2 (fixed)
def f_jpeg_grid(rgb, gray, fmt):
    if fmt != 'PNG':
        return 0.0, None
    gx = np.abs(np.diff(gray, axis=1))
    hb, wb = gx.shape[0] // B, gx.shape[1] // B          # derive grid from gx
    if hb < 4 or wb < 4:
        return 0.0, None
    g = gx[:hb * B, :wb * B].reshape(hb, B, wb, B)
    seam = (g[:, 7, :, :] + g[:, 15, :, :]).mean(axis=2) / 2.0
    inner = g[:, :7, :, :7].mean(axis=(1, 3))
    ratio = seam / (inner + 0.35)
    var = g.var(axis=(1, 3))
    smooth = var < np.percentile(var, 55)                # judge only flat areas
    if smooth.sum() < 20:
        return 0.0, None
    z = np.where(smooth, ratio - np.median(ratio[smooth]), 0.0)
    # contiguous rectangle of high-seam smooth blocks
    hot = smooth & (z > np.percentile(z[smooth], 97))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
    best, box = 0.0, None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        sc = min(1.0, area / 24.0) * float(z[lab == i].mean() / 2.0)
        if sc > best:
            best, box = min(sc, 1.0), (x * B, y * B, (x + w) * B, (y + h) * B)
    return best, box


# ----------------------------------------------------------------- F14 (block)
def f_foreign_patch(rgb, gray, fmt):
    keys = ckeys(rgb)
    u, c = np.unique(keys.ravel(), return_counts=True)
    tot = keys.size
    pal = set(u[c >= max(180, tot * 0.00035)].tolist())   # the document's design palette
    off = ~np.isin(keys.ravel(), list(pal)).reshape(keys.shape)
    hb, wb = off.shape[0] // B, off.shape[1] // B
    if hb < 4 or wb < 4:
        return 0.0, None
    frac = off[:hb * B, :wb * B].reshape(hb, B, wb, B).mean(axis=(1, 3))
    hot = frac > 0.45
    if hot.sum() < 6:
        return 0.0, None
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
    best, box = 0.0, None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 6 or w * h > area * 3.2:                # must be rectangle-like
            continue
        sc = min(1.0, area / 40.0) * float(frac[lab == i].mean())
        if sc > best:
            best, box = sc, (x * B, y * B, (x + w) * B, (y + h) * B)
    return best, box


# ----------------------------------------------------------------- F13 (tight)
def f_glyph_drift(rgb, gray, fmt):
    mask = ink_mask(gray)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    gl = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if 7 <= h <= 55 and 2 <= w <= 60 and 16 <= area <= w * h * 0.9:
            sel = lab == i
            if sel.sum() >= 14:
                gl.append((int(x), int(y), int(w), int(h), rgb[sel].mean(axis=0)))
    if len(gl) < 30:
        return 0.0, None
    cols = np.array([g[4] for g in gl])
    q = (cols // 14).astype(int)
    kk = (q[:, 0] << 20) | (q[:, 1] << 10) | q[:, 2]
    vals, cnt = np.unique(kk, return_counts=True)
    design = set(vals[cnt >= max(3, len(gl) * 0.05)].tolist())
    modal = vals[cnt.argmax()]
    modal_rgb = dec(modal)
    off = []
    for (x, y, w, h, c), kv in zip(gl, kk):
        if kv in design:
            continue
        d = float(np.sqrt(((c - modal_rgb) ** 2).sum()))
        if 14.0 < d < 45.0:
            off.append((x, y, w, h))
    if len(off) < 4:
        return 0.0, None
    off.sort(key=lambda t: (t[1] // 10, t[0]))
    run, best, box, cur = 1, 0.0, None, [off[0]]
    for i in range(1, len(off)):
        p, c2 = off[i - 1], off[i]
        if abs(c2[1] - p[1]) <= 12 and 0 <= c2[0] - (p[0] + p[2]) <= 34:
            run += 1; cur.append(c2)
        else:
            run, cur = 1, [c2]
        if run > best:
            best = run
            xs = [t[0] for t in cur]; ys = [t[1] for t in cur]
            box = (min(xs), min(ys), max(t[0] + t[2] for t in cur), max(t[1] + t[3] for t in cur))
    return float(min(best, 10) / 10.0), box


# ------------------------------------------------------------- F8 (unchanged)
def f_white_mark(rgb, gray, fmt):
    keys = ckeys(rgb)
    u, c = np.unique(keys.ravel(), return_counts=True)
    tot = keys.size
    best, box = 0.0, None
    for k, n in zip(u, c):
        if n < 300 or n > tot * 0.09:
            continue
        m = keys == k
        ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if len(ys) < 6 or len(xs) < 6:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        fill = n / ((y1 - y0 + 1) * (x1 - x0 + 1))
        if fill < 0.82:
            continue
        h, w = keys.shape
        parts = [rgb[max(0, y0 - 9):max(0, y0 - 2), x0:x1 + 1], rgb[min(h, y1 + 3):min(h, y1 + 10), x0:x1 + 1],
                 rgb[y0:y1 + 1, max(0, x0 - 9):max(0, x0 - 2)], rgb[y0:y1 + 1, min(w, x1 + 3):min(w, x1 + 10)]]
        ring = [p.reshape(-1, 3) for p in parts if p.size]
        if not ring:
            continue
        d = float(np.sqrt(((dec(k) - np.median(np.concatenate(ring).astype(float), axis=0)) ** 2).sum()))
        if 1.5 < d < 34.0:
            sc = fill * min(1.0, n / 2000.0)
            if sc > best:
                best, box = sc, (int(x0), int(y0), int(x1), int(y1))
    return best, box


# ------------------------------------------------------------------ F15 clone
def f_orb_clone(rgb, gray, fmt):
    g8 = cv2.resize(gray.astype(np.uint8), None, fx=0.5, fy=0.5)
    orb = cv2.ORB_create(nfeatures=1600, fastThreshold=6)
    kp, des = orb.detectAndCompute(g8, None)
    if des is None or len(kp) < 40:
        return 0.0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    offs, pts = [], []
    for i, ms in enumerate(bf.radiusMatch(des, des, 6)):
        for m in ms:
            if m.queryIdx == m.trainIdx:
                continue
            a, b = kp[m.queryIdx].pt, kp[m.trainIdx].pt
            d = (b[0] - a[0], b[1] - a[1])
            if 24 < (d[0] ** 2 + d[1] ** 2) ** 0.5 < 900:
                offs.append((round(d[0] / 3), round(d[1] / 3))); pts.append((a[0] + b[0], a[1] + b[1]))
    if len(offs) < 12:
        return 0.0, None
    vals, cnt = np.unique(np.array(offs), axis=0, return_counts=True)
    top = int(cnt.max())
    box = None
    if top >= 12:
        v = vals[cnt.argmax()]
        sel = np.array([p for p, o in zip(pts, offs) if tuple(o) == (v[0], v[1])]) * 2
        box = (int(sel[:, 0].min()), int(sel[:, 1].min()), int(sel[:, 0].max()), int(sel[:, 1].max()))
    return float(np.clip((top - 10) / 45.0, 0, 1)), box


# ------------------------------------------------- F16 resampled (soft) patch
def f_soft_patch(rgb, gray, fmt):
    """A pasted patch is almost always RESIZED, so its edges are interpolator-soft
    while the rest of the document is natively crisp. Score the largest rectangle
    of anomalously soft blocks."""
    gxm = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gym = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gxm, gym)
    hb, wb = mag.shape[0] // B, mag.shape[1] // B
    if hb < 4 or wb < 4:
        return 0.0, None
    m = mag[:hb * B, :wb * B].reshape(hb, B, wb, B)
    ink = m.max(axis=(1, 3))                       # peak edge energy per block
    mass = m.mean(axis=(1, 3))                     # average edge energy per block
    with np.errstate(divide='ignore', invalid='ignore'):
        crisp = np.where(ink > 30, mass / ink, np.nan)   # low = sharp, high = soft
    valid = ~np.isnan(crisp)
    if valid.sum() < 30:
        return 0.0, None
    thr = np.nanpercentile(crisp, 96)
    hot = valid & (crisp > thr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
    best, box = 0.0, None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 8 or w * h > area * 2.6:
            continue
        sc = min(1.0, area / 30.0) * min(1.0, (float(np.nanmean(crisp[lab == i])) / max(thr, 1e-6) - 0.85) * 3.0)
        if sc > best:
            best, box = float(np.clip(sc, 0, 1)), (x * B, y * B, (x + w) * B, (y + h) * B)
    return best, box


FEATURES = [('F16_soft_patch', f_soft_patch), ('F8_white_mark', f_white_mark), ('F2_jpeg_grid', f_jpeg_grid),
            ('F14_foreign_patch', f_foreign_patch), ('F13_glyph_drift', f_glyph_drift),
            ('F15_orb_clone', f_orb_clone)]
TH = {'F16_soft_patch': 0.30, 'F8_white_mark': 0.35, 'F2_jpeg_grid': 0.30, 'F14_foreign_patch': 0.30,
      'F13_glyph_drift': 0.40, 'F15_orb_clone': 0.35}


def iou(a, b):
    if not a or not b or b == 'None':
        return 0.0
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    rows = []
    for m in MAN:
        rgb, fmt = load(m['path'])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        bb = m.get('bbox')
        bb = tuple(bb) if isinstance(bb, (list, tuple)) and len(bb) == 4 else None
        rec = {'path': m['path'], 'label': m['label'], 'op': str(m.get('op')), 'bbox': bb}
        for n, fn in FEATURES:
            try:
                v, box = fn(rgb, gray, fmt)
                rec[n] = float(v)
                rec[n + '_iou'] = iou(box, bb) if bb else 0.0
            except Exception as e:
                rec[n] = float('nan'); rec[n + '_err'] = f"{type(e).__name__}: {e}"[:110]
        rows.append(rec)

    gen = [r for r in rows if r['label'] == 'genuine']
    tam = [r for r in rows if r['label'] == 'tampered']
    ops = sorted({r['op'] for r in tam})
    names = [n for n, _ in FEATURES]
    print(f"corpus {len(gen)} genuine / {len(tam)} tampered  ops={ops}\n" + "=" * 104)
    print("TEST                   THRESH   GEN false-pos   CATCH RATE PER OP                       mean IoU")
    print("=" * 104)
    for n in names:
        t = TH[n]
        fp = sum(1 for r in gen if r[n] >= t)
        per = []
        for op in ops:
            sub = [r for r in tam if r['op'] == op]
            per.append(f"{op[:5]}:{sum(1 for r in sub if r[n] >= t)}/{len(sub)}")
        hits = [r for r in tam if r[n] >= t and r['bbox']]
        mi = float(np.mean([r[n + '_iou'] for r in hits])) if hits else 0.0
        print(f"{n:22s} {t:5.2f}   {fp:2d}/{len(gen)}          {' '.join(per):38s} {mi:6.3f}")
    print("=" * 104)
    any_det = [r for r in tam if any(r[n] >= TH[n] for n in names)]
    any_fp = [r for r in gen if any(r[n] >= TH[n] for n in names)]
    print(f"\nOR-of-tests: catches {len(any_det)}/{len(tam)} tampered, false-flags {len(any_fp)}/{len(gen)} genuine")
    miss = {}
    for r in tam:
        if not any(r[n] >= TH[n] for n in names):
            miss[r['op']] = miss.get(r['op'], 0) + 1
    print("missed by op:", miss or "none")
    e = {k: r[k] for r in rows for k in r if k.endswith('_err')}
    if e:
        print("ERRORS:", list(e.items())[:3])
    json.dump({'rows': rows}, open('models/feature_probe.json', 'w'), default=str)


if __name__ == '__main__':
    main()
