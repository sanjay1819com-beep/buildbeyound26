#!/usr/bin/env python3
"""
Builds a GROUND-TRUTH image-forensics calibration corpus.

The first version of this engine scored a genuine, never-edited screenshot at 67.6%
"LIKELY MANIPULATED". That is because its thresholds were guessed, and a UI screenshot
legitimately contains everything a naive tamper test looks for: perfectly straight
axis-aligned edges, repeated bubbles, flat background next to crisp text.

So: synthesise genuine screenshots, apply KNOWN tampering operations at KNOWN locations,
then measure how well each test separates the two populations and set thresholds from
the data. Ground truth also gives us an honest accuracy figure to show a judge instead
of an invented one.

Tamper operations modelled on what actually happens in Indian cyberstalking / fraud cases:
  eraser      rub out a line of a chat and leave the background fill  (the "white mark")
  overtype    erase a figure and type a different one over it
  amount      change a payment amount  (₹5,000 -> ₹85,000)
  paste       splice a rectangle in from a different screenshot
  clone       copy-move an existing bubble elsewhere in the same image
"""
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

OUT = "forensic_corpus"
W, H = 720, 980
LAYOUTS = 12


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


F_H = font(19)
F_B = font(16)
F_S = font(13)
F_T = font(15)

PALETTES = [
    {"bg": "#0f1420", "hdr": "#1b2434", "me": "#1f6feb", "them": "#242d3d",
     "txt": "#e8eef7", "sub": "#8b98ab", "accent": "#22d3ee"},
    {"bg": "#f4f6f8", "hdr": "#ffffff", "me": "#d9fdd3", "them": "#ffffff",
     "txt": "#111b21", "sub": "#667781", "accent": "#00a884"},
    {"bg": "#101018", "hdr": "#1c1c28", "me": "#3b2f63", "them": "#23232f",
     "txt": "#f0eefb", "sub": "#9a94b8", "accent": "#a78bfa"},
    {"bg": "#ffffff", "hdr": "#f0f2f5", "me": "#e7f3ff", "them": "#f2f3f5",
     "txt": "#050505", "sub": "#65676b", "accent": "#1877f2"},
    {"bg": "#0b1220", "hdr": "#131c2e", "me": "#0e4429", "them": "#202c33",
     "txt": "#e9edef", "sub": "#8696a0", "accent": "#34d399"},
]

CHATS = [
    [("Riya", "Please stop messaging me. I have told HR."),
     ("+91 98765 43210", "Blocking me will not help. Transfer the money tonight."),
     ("Riya", "This is extortion and stalking. Leave me alone."),
     ("+91 98765 43210", "Saw you leaving Prestige Tech Park Gate 2 at 9:30 PM."),
     ("Riya", "I am recording every message."),
     ("+91 91234 56789", "New number, same warning. Approve the collect request."),
     ("Riya", "Why are you following me?"),
     ("+91 91234 56789", "Last chance. Do not test my patience.")],
    [("Support", "Your payment of Rs 5,000 was received successfully."),
     ("Me", "I did not authorise any payment."),
     ("Support", "Reference UPI6283910441 linked to HDFC account 4491."),
     ("Me", "Please reverse it immediately."),
     ("Support", "A collect request for Rs 85,000 is pending approval."),
     ("Me", "I am reporting this to the cyber cell."),
     ("Support", "Your complaint has been logged."),
     ("Me", "Send me the acknowledgement number.")],
    [("Ankit", "Did you see the post on the office group?"),
     ("Me", "No. What happened?"),
     ("Ankit", "Someone uploaded a morphed photo with your badge."),
     ("Me", "Which account posted it?"),
     ("Ankit", "The handle is shadow_watch_blr."),
     ("Me", "I have never heard of that account."),
     ("Ankit", "It was created last week from a Bengaluru IP."),
     ("Me", "I am taking a screenshot of this.")],
    [("Bank", "OTP for your transaction is 444921. Do not share."),
     ("Me", "I did not request any OTP."),
     ("Bank", "A collect request was raised on your VPA."),
     ("Me", "Block my UPI handle immediately."),
     ("Bank", "Request declined by you at 20:05 IST."),
     ("Me", "Who raised the request?"),
     ("Bank", "Requester VPA rohan.v88 at okicici."),
     ("Me", "That is not a person I know.")],
    [("Neha", "He called me asking where you sit."),
     ("Me", "What did you tell him?"),
     ("Neha", "Nothing. But he knows your shift timing."),
     ("Me", "He has been outside the gate twice."),
     ("Neha", "Should we inform security?"),
     ("Me", "I already filed a complaint online."),
     ("Neha", "Keep the acknowledgement safe."),
     ("Me", "Everything is backed up and hashed.")],
]


def genuine(idx, seed=0):
    pal = PALETTES[idx % len(PALETTES)]
    R = random.Random(seed * 977 + idx)
    img = Image.new("RGB", (W, H), pal["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 96], fill=pal["hdr"])
    d.ellipse([20, 26, 68, 74], fill=pal["accent"])
    d.text((82, 30), "Conversation", font=F_H, fill=pal["txt"])
    d.text((82, 58), "online · last seen today at 21:44", font=F_S, fill=pal["sub"])
    d.text((W - 150, 40), "21:44", font=F_S, fill=pal["sub"])
    y = 120
    boxes = []
    for i, (who, txt) in enumerate(CHATS[idx % len(CHATS)]):
        mine = who in ("Me", "Riya")
        bw = min(W - 190, 46 + len(txt) * 8 + R.randint(-14, 26))
        x0 = (W - bw - 22 + R.randint(-6, 6)) if mine else (22 + R.randint(-6, 6))
        lines = 1 + len(txt) // 42
        bh = 34 + lines * 22 + R.randint(-3, 6)
        d.rounded_rectangle([x0, y, x0 + bw, y + bh], radius=14,
                            fill=pal["me"] if mine else pal["them"])
        d.text((x0 + 14, y + 9), txt[:44], font=F_B, fill=pal["txt"])
        if len(txt) > 44:
            d.text((x0 + 14, y + 31), txt[44:], font=F_B, fill=pal["txt"])
        d.text((x0 + bw - 58, y + bh - 20), "21:%02d" % (10 + i), font=F_S, fill=pal["sub"])
        boxes.append((x0, y, x0 + bw, y + bh, mine))
        y += bh + 16 + R.randint(-2, 5)
        if y > H - 120:
            break
    d.rectangle([0, H - 66, W, H], fill=pal["hdr"])
    d.rounded_rectangle([16, H - 52, W - 76, H - 16], radius=18, fill=pal["them"])
    d.text((32, H - 44), "Message", font=F_T, fill=pal["sub"])
    d.ellipse([W - 62, H - 52, W - 22, H - 12], fill=pal["accent"])
    return img, boxes, pal


def rng(seed):
    return random.Random(seed)


def tamper_eraser(img, boxes, pal):
    """Rub out a whole message line, leaving the flat background fill (the white mark)."""
    im = img.copy(); d = ImageDraw.Draw(im)
    cands = [b for b in boxes if b[3] - b[1] > 50 and 140 < b[1] < H - 200]
    if not cands:
        return None
    x0, y0, x1, y1, mine = rng(7).choice(cands)
    # fill with the *bubble* colour to mimic a lazy erase, slightly off so a halo remains
    base = Image.new("RGB", (1, 1), pal["me"] if mine else pal["them"]).getpixel((0, 0))
    off = tuple(min(255, c + 9) for c in base)
    d.rectangle([x0 + 10, y0 + 6, x1 - 62, y1 - 8], fill=off)
    return im, [x0 + 10, y0 + 6, x1 - 62, y1 - 8], "eraser"


def tamper_overtype(img, boxes, pal):
    """Erase a figure and type a different one over it."""
    im = img.copy(); d = ImageDraw.Draw(im)
    cands = [b for b in boxes if b[3] - b[1] > 50 and 140 < b[1] < H - 200]
    if not cands:
        return None
    x0, y0, x1, y1, mine = rng(11).choice(cands)
    base = Image.new("RGB", (1, 1), pal["me"] if mine else pal["them"]).getpixel((0, 0))
    d.rectangle([x0 + 12, y0 + 7, x0 + 210, y0 + 32], fill=base)
    d.text((x0 + 14, y0 + 10), "Rs 85,000 transferred now", font=F_B,
           fill=tuple(min(255, c + 26) for c in Image.new("RGB", (1, 1), pal["txt"]).getpixel((0, 0))))
    return im, [x0 + 12, y0 + 7, x0 + 210, y0 + 32], "overtype"


def tamper_paste(img, boxes, pal):
    """Splice a rectangle in from a DIFFERENT screenshot (different palette/renderer)."""
    im = img.copy()
    other, ob, opal = genuine((PALETTES.index(pal) + 2) % len(PALETTES), seed=99)
    src = other.crop((40, 300, 300, 400)).resize((250, 92))
    x, y = 60, H - 330
    im.paste(src, (x, y))
    return im, [x, y, x + 250, y + 92], "paste"


def tamper_clone(img, boxes, pal):
    """Copy-move: lift an existing bubble and stamp it lower down."""
    im = img.copy()
    cands = [b for b in boxes if b[3] - b[1] > 50 and 140 < b[1] < H - 320]
    if not cands:
        return None
    x0, y0, x1, y1, _ = rng(3).choice(cands)
    patch = im.crop((x0, y0, x1, y1))
    tx, ty = x0 + 26, min(y1 + 60, H - 150 - (y1 - y0))
    im.paste(patch, (tx, ty))
    return im, [tx, ty, tx + (x1 - x0), ty + (y1 - y0)], "clone"


def tamper_jpeg(img, boxes, pal):
    """Re-save one region through JPEG then composite back (double-compression splice)."""
    im = img.copy()
    x0, y0, x1, y1 = 40, 150, 400, 330
    region = im.crop((x0, y0, x1, y1))
    import io as _io
    buf = _io.BytesIO(); region.save(buf, "JPEG", quality=42)
    buf.seek(0)
    im.paste(Image.open(buf).convert("RGB"), (x0, y0))
    return im, [x0, y0, x1, y1], "jpeg_splice"


OPS = [tamper_eraser, tamper_overtype, tamper_paste, tamper_clone, tamper_jpeg]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "genuine"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "tampered"), exist_ok=True)
    manifest = []
    for i in range(LAYOUTS):
        img, boxes, pal = genuine(i, seed=i)
        gp = os.path.join(OUT, "genuine", f"g{i}.png")
        img.save(gp)
        manifest.append({"path": gp, "label": "genuine", "op": None, "bbox": None})
        for op in OPS:
            r = op(img, boxes, pal)
            if not r:
                continue
            tim, bbox, name = r
            tp = os.path.join(OUT, "tampered", f"g{i}_{name}.png")
            tim.save(tp)
            manifest.append({"path": tp, "label": "tampered", "op": name, "bbox": bbox})
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    g = sum(1 for m in manifest if m["label"] == "genuine")
    print(f"corpus: {g} genuine + {len(manifest)-g} tampered = {len(manifest)} images -> {OUT}/")
    from collections import Counter
    print("  ops:", dict(Counter(m["op"] for m in manifest if m["op"])))


if __name__ == "__main__":
    main()
