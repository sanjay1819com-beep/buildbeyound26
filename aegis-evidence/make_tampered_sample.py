"""Create the doctored exhibit used in the live demo.

STORY: EV-003 is the victim's own screenshot of the conversation. EV-008 is the
version the ACCUSED produced - the same conversation with the extortion demand
("pay Rs 85,000 ...") erased. Erasing is the single most common amateur forgery,
and it leaves exactly the artefact a lay reviewer notices by eye: a rectangle of
*almost* the right colour. That is the "white mark".

The erase is placed by MEASURING the bubble, then shrunk until all four sides of
the patch are surrounded by pure bubble colour - so the detector fires because
the artefact is really there, not because the rectangle happened to land on a
boundary. The embedded OCR text is rebuilt without the erased line, which is what
makes the two versions of the same conversation contradict each other.
"""
import io
import os
import numpy as np
import cv2
from PIL import Image, PngImagePlugin

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sample_evidence/03_instagram_dm_screenshot.png")
DST = os.path.join(HERE, "sample_evidence/08_doctored_instagram_dm_screenshot.png")
OFFSET = 9                      # the amateur's "almost right" fill
ERASE_OCR_LINES = [2]           # the extortion demand


def bubble_of(rgb, gray, y0, y1, x0, x1):
    """Bounds of the speech bubble that contains the given text band."""
    seed = rgb[(y0 + y1) // 2, (x0 + x1) // 2].astype(float)
    d = np.sqrt(((rgb.astype(float) - seed) ** 2).sum(axis=2))
    same = d < 9
    lab = np.zeros(same.shape, np.int32)
    n, lab, st, _ = cv2.connectedComponentsWithStats(same.astype(np.uint8), 8)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    best, bb = 0, None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area > best and x <= cx < x + w and y <= cy < y + h:
            best, bb = area, (x, y, x + w - 1, y + h - 1)
    return bb, seed


def main():
    im = Image.open(SRC)
    rgb = np.array(im.convert("RGB")).astype(np.float64)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    loc = cv2.medianBlur(gray.astype(np.uint8), 31).astype(np.float64)
    ink = np.abs(gray - loc) > 52
    ink[:60, :] = False
    ink[835:, :] = False
    rows = np.flatnonzero(ink.sum(axis=1) > 10)
    bands = []
    for y in rows:
        if bands and y - bands[-1][-1] <= 4:
            bands[-1].append(y)
        else:
            bands.append([y])
    # the extortion demand is the suspect-side block of three consecutive text lines
    tgt = [b for b in bands if b[0] > 320 and b[-1] < 400]
    assert tgt, "could not locate the extortion demand text bands"
    ty0, ty1 = tgt[0][0], tgt[-1][-1]
    ys = list(range(ty0, ty1 + 1))
    tx0 = int(np.flatnonzero(ink[ys].any(axis=0)).min())
    tx1 = int(np.flatnonzero(ink[ys].any(axis=0)).max())
    bb, seed = bubble_of(rgb, gray, ty0, ty1, tx0, tx1)
    assert bb, "could not locate the containing speech bubble"
    bx0, by0, bx1, by1 = bb
    print(f"text bands y {ty0}-{ty1} x {tx0}-{tx1}   bubble {bb} colour {seed.astype(int)}")

    fill = np.clip(seed + OFFSET, 0, 255)
    out = rgb.copy()
    # shrink until every side of the patch is surrounded by PURE bubble colour
    for inset in range(14, 90, 4):
        px0, py0 = max(bx0 + inset, tx0 - 18), max(by0 + inset, ty0 - 10)
        px1, py1 = min(bx1 - inset, tx1 + 18), min(by1 - inset, ty1 + 10)
        if px1 - px0 < 60 or py1 - py0 < 20:
            continue
        sides = [out[py0 - 9:py0 - 2, px0:px1 + 1], out[py1 + 3:py1 + 10, px0:px1 + 1],
                 out[py0:py1 + 1, px0 - 9:px0 - 2], out[py0:py1 + 1, px1 + 3:px1 + 10]]
        if any(not s.size for s in sides):
            continue
        # purity = each side matches the BUBBLE colour (not the fill: the fill is
        # deliberately offset by sqrt(9^2*3)=15.6, which is the whole point)
        deltas = [float(np.sqrt(((np.median(s.reshape(-1, 3), axis=0) - seed) ** 2).sum())) for s in sides]
        if max(deltas) <= 10.0:
            print(f"patch ({px0},{py0})-({px1},{py1})  fill {fill.astype(int)}  "
                  f"surround deltas {[round(d,1) for d in deltas]}  -> accepted")
            out[py0:py1 + 1, px0:px1 + 1] = fill
            box = (px0, py0, px1, py1)
            break
    else:
        raise SystemExit("could not place a patch with a pure surround")

    # rebuild the embedded OCR text without the erased line, exactly as a doctored
    # screenshot would read if it were re-OCR'd
    lines = im.info.get("ExtractedScreenText", "").split("\n")
    kept = [l for i, l in enumerate(lines) if i not in ERASE_OCR_LINES]
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("ExtractedScreenText", "\n".join(kept))
    pnginfo.add_text("CaptureDevice", im.info.get("CaptureDevice", "iPhone 15 Pro iOS 19.1"))
    pnginfo.add_text("CaptureTimestampIST", im.info.get("CaptureTimestampIST", ""))
    pnginfo.add_text("ProvenanceNote", "Received from the accused's counsel by email; "
                                       "offered as the complete conversation.")
    Image.fromarray(out.astype(np.uint8)).save(DST, "PNG", pnginfo=pnginfo)
    print(f"wrote {os.path.relpath(DST, HERE)}  ({os.path.getsize(DST)} bytes)")
    print(f"erased OCR line(s): {[lines[i][:70] for i in ERASE_OCR_LINES]}")

    # ---- verify the detector actually catches it -----------------------------
    import image_forensics as FX
    r = FX.analyse_image(open(DST, "rb").read())
    g = FX.analyse_image(open(SRC, "rb").read())
    print(f"\nDOCTORED EV-008 -> {r['verdict']} {r['fusion_pct']}%   bbox={r['primary_region']}")
    print(f"  ground truth box we painted: {box}")
    for k, v in sorted(r["tests"].items(), key=lambda kv: -kv[1]["score"]):
        if v["score"] > 0:
            print(f"   {k:22s} {v['score_pct']:5.1f}%  {v['detail'][:110]}")
    print(f"\nORIGINAL EV-003 -> {g['verdict']} {g['fusion_pct']}%   (must stay clean)")
    assert g["fusion_pct"] < 22, "the genuine original must not be flagged"
    assert r["fusion_pct"] >= 40, f"the doctored copy must be flagged, got {r['fusion_pct']}%"
    if r["primary_region"]:
        x0, y0, x1, y1 = r["primary_region"]
        ix = max(0, min(x1, box[2]) - max(x0, box[0])) * max(0, min(y1, box[3]) - max(y0, box[1]))
        ua = (x1 - x0) * (y1 - y0) + (box[2] - box[0]) * (box[3] - box[1]) - ix
        print(f"localisation IoU vs the true edit box: {ix/ua:.3f}")
    print("\nOK: doctored exhibit is caught, genuine original is clean.")


if __name__ == "__main__":
    main()
