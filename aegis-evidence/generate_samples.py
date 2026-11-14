#!/usr/bin/env python3
"""
Generates realistic multi-modal sample evidence files for PS 3.2 Demo:
Case: State / Riya Sharma vs. Unknown Cyber-Harasser & Fraudster ("Rohan Verma" / Alias Accounts)
"""
import os
import wave
import math
import struct
import json
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import docx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "sample_evidence")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. WhatsApp Chat Export (.txt)
whatsapp_chat = """[10/10/2026, 09:14:22] Riya Sharma: Please stop messaging me. I already told HR and blocked your previous number.
[10/10/2026, 09:15:04] +91 98765 43210: Blocking me won't help Riya. Remember my name - Rohan. If you don't transfer ₹85,000 to UPI id rohan.v88@okicici by tonight, your morphed photos go straight to your company board.
[10/10/2026, 09:16:30] Riya Sharma: This is extortion and stalking. Leave me alone!
[10/10/2026, 21:42:11] +91 98765 43210: Saw you leaving Prestige Tech Park Gate 2 at 9:30 PM in your blue Swift. Wearing the white jacket? Don't test me.
[12/10/2026, 14:05:19] +91 91234 56789: Hey Riya, new number, same warning. Send ₹85,000 to rohan.v88@okicici or check Instagram handle @shadow_watch_blr.
[14/10/2026, 19:20:00] +91 91234 56789: Why did you talk to Inspector Nair? Big mistake. Watch your inbox tonight at 11 PM.
"""
with open(os.path.join(OUT_DIR, "01_whatsapp_chat_export.txt"), "w", encoding="utf-8") as f:
    f.write(whatsapp_chat)

# 2. Raw Threatening Email (.eml) - WITH AN INTENTIONAL HEADER SPOOFING / INCONSISTENCY!
# Notice: Displayed Date claims 08 Oct 2026 (alibi attempt / backdated client clock),
# while SMTP Received header proves server received it on 14 Oct 2026 23:04:12 IST,
# AND sender claims to be in Mumbai while X-Originating-IP resolves to Bengaluru ISP!
raw_email = """Return-Path: <shadow.blr.ops@protonmail.com>
Received: from mail-relay.protonmail.ch (mail-relay.protonmail.ch [185.70.40.101])
        by mx.google.com with ESMTPS id k19-20261014230412
        for <riya.sharma@techcorp.in>;
        Wed, 14 Oct 2026 23:04:12 +0530 (IST)
X-Originating-IP: [49.207.212.88] (ACT Fibernet - Koramangala, Bengaluru, KA)
From: "Anonymous Vigilante" <shadow.blr.ops@protonmail.com>
Reply-To: rohan.verma.official@gmail.com
To: riya.sharma@techcorp.in
Subject: FINAL DEMAND - Pay Rs 85,000 or Facing Consequences
Date: Thu, 08 Oct 2026 11:00:00 +0530
Message-ID: <9982341a-bc82@protonmail.com>
MIME-Version: 1.0
Content-Type: text/plain; charset="UTF-8"

Riya,

I am writing this from Mumbai so Bengaluru police can't touch me.
Just like I warned you after you spoke to Inspector Nair today, this is your last chance.
Transfer Rs. 85,000 to UPI: rohan.v88@okicici (Account holder: R. Verma, HDFC A/c ending 4491).
Check Instagram @shadow_watch_blr — I already posted your office badge photo.
Delete this email if you want, nobody will believe you.

- R.V.
"""
with open(os.path.join(OUT_DIR, "02_threatening_email_raw.eml"), "w", encoding="utf-8") as f:
    f.write(raw_email)

# 3. Screenshot 1: Instagram Stalking / Impersonation Threat (.png) with embedded OCR metadata & clean visual
def create_screenshot(filename, header_title, handle, timestamp_str, messages, badge_color=(225, 48, 108), tamper_note=None):
    img = Image.new("RGB", (720, 900), color=(15, 18, 26))
    draw = ImageDraw.Draw(img)
    # Top phone status bar
    draw.rectangle([0, 0, 720, 44], fill=(10, 12, 18))
    draw.text((24, 14), "22:18  5G", fill=(220, 225, 235))
    draw.text((620, 14), "84%", fill=(120, 230, 160))

    # App header
    draw.rectangle([0, 44, 720, 124], fill=(24, 29, 42))
    draw.ellipse([24, 60, 76, 112], fill=badge_color)
    draw.text((42, 78), "SW", fill=(255, 255, 255))
    draw.text((92, 64), header_title, fill=(255, 255, 255))
    draw.text((92, 88), handle, fill=(150, 165, 190))

    # Timestamp pill
    draw.rounded_rectangle([230, 145, 490, 178], radius=14, fill=(32, 38, 54))
    draw.text((255, 155), timestamp_str, fill=(180, 195, 220))

    y = 205
    for sender, text, is_me in messages:
        box_w = 520
        x0 = 170 if is_me else 30
        x1 = x0 + box_w
        lines = []
        words = text.split(" ")
        cur = ""
        for w in words:
            if len(cur + " " + w) > 52:
                lines.append(cur.strip())
                cur = w
            else:
                cur += " " + w
        if cur.strip():
            lines.append(cur.strip())
        box_h = 46 + len(lines) * 24
        fill_col = (37, 99, 235) if is_me else (34, 41, 58)
        draw.rounded_rectangle([x0, y, x1, y + box_h], radius=16, fill=fill_col)
        draw.text((x0 + 18, y + 12), sender, fill=(147, 197, 253) if not is_me else (219, 234, 254))
        ty = y + 36
        for ln in lines:
            draw.text((x0 + 18, ty), ln, fill=(245, 247, 250))
            ty += 24
        y += box_h + 22

    # Footer forensic capture banner
    draw.rectangle([0, 830, 720, 900], fill=(18, 24, 36))
    draw.text((24, 845), f"Captured on Riya's iPhone 15 | Source: {handle}", fill=(130, 150, 180))
    if tamper_note:
        draw.text((24, 868), tamper_note, fill=(251, 191, 36))
    else:
        draw.text((24, 868), "EXIF Timestamp: 2026-10-12 22:18:40 IST | Original PNG", fill=(94, 234, 212))

    path = os.path.join(OUT_DIR, filename)
    meta = PngImagePlugin.PngInfo()
    ocr_lines = [f"{header_title} ({handle}) - {timestamp_str}"] + [f"{s}: {t}" for s, t, _ in messages]
    meta.add_text("ExtractedScreenText", "\n".join(ocr_lines))
    meta.add_text("CaptureDevice", "iPhone 15 Pro iOS 19.1")
    meta.add_text("CaptureTimestampIST", "2026-10-12T22:18:40+05:30")
    img.save(path, format="PNG", pnginfo=meta)

create_screenshot(
    "03_instagram_dm_screenshot.png",
    "Shadow Watch BLR",
    "@shadow_watch_blr",
    "12 Oct 2026, 10:15 PM",
    [
        ("@shadow_watch_blr", "Nice blue Swift parked near Prestige Tech Park Gate 2 on Oct 10.", False),
        ("@shadow_watch_blr", "Tell Rohan Verma sorry and pay Rs 85,000 to rohan.v88@okicici. Phone +91 91234 56789 is waiting.", False),
        ("Riya Sharma", "I know @shadow_watch_blr is you, Rohan! Stop stalking me or I am filing an FIR.", True),
        ("@shadow_watch_blr", "Prove it's me :) Accounts vanish in seconds.", False),
    ]
)

# 4. Near-Duplicate / Cropped & Re-uploaded Screenshot (.png) to demonstrate Perceptual Hashing (pHash/dHash) Duplicate Detection!
orig_img = Image.open(os.path.join(OUT_DIR, "03_instagram_dm_screenshot.png"))
# Slightly crop 4 pixels & re-save without metadata to simulate WhatsApp forwarded / re-saved duplicate
cropped = orig_img.crop((2, 2, 718, 898)).resize((720, 900))
dup_meta = PngImagePlugin.PngInfo()
dup_meta.add_text("ExtractedScreenText", orig_img.info.get("ExtractedScreenText", ""))
dup_meta.add_text("CaptureDevice", "WhatsApp Forwarded Re-Save (Stripped Original EXIF)")
dup_meta.add_text("CaptureTimestampIST", "2026-10-15T08:10:00+05:30")
cropped.save(os.path.join(OUT_DIR, "04_forwarded_duplicate_ig_screenshot.png"), format="PNG", pnginfo=dup_meta)

# 5. Voicemail Audio Recording (.wav) with realistic speech-cadence waveform + companion forensic transcript metadata
wav_path = os.path.join(OUT_DIR, "05_threatening_voicemail_oct13.wav")
sample_rate = 16000
duration_sec = 4.5
num_samples = int(sample_rate * duration_sec)
with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    frames = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        # Modulated voice-like formants + pauses so audio player plays an audible signal & waveform looks realistic
        envelope = 0.0 if (1.1 < t < 1.35 or 2.7 < t < 2.95) else (0.5 + 0.4 * math.sin(2 * math.pi * 3.5 * t))
        val = envelope * (
            0.55 * math.sin(2 * math.pi * 165 * t) +
            0.30 * math.sin(2 * math.pi * 330 * t) +
            0.15 * math.sin(2 * math.pi * 820 * t)
        )
        sample = max(-32767, min(32767, int(val * 14000)))
        frames.extend(struct.pack("<h", sample))
    wf.writeframes(bytes(frames))

# 6. Suspect's HR / Police Statement (.docx) — Contains DIRECT CONTRADICTIONS against digital evidence!
doc = docx.Document()
doc.add_heading("WITNESS / RESPONDENT STATEMENT — INTERNAL & POLICE INQUIRY", 0)
doc.add_paragraph("Date of Statement: 15 October 2026")
doc.add_paragraph("Respondent Name: Rohan Verma (Ex-Contractor, TechCorp Bengaluru)")
doc.add_paragraph("Registered Phone: +91 98765 43210 | Secondary Email: rohan.verma.official@gmail.com")
doc.add_heading("Recorded Declaration:", level=2)
doc.add_paragraph(
    "1. I completely deny sending any messages or threats to Ms. Riya Sharma after 05 October 2026. "
    "I have had zero contact with her in October 2026."
)
doc.add_paragraph(
    "2. From 09 October 2026 to 15 October 2026, I was physically in Mumbai attending a family event "
    "and was nowhere near Bengaluru or Prestige Tech Park."
)
doc.add_paragraph(
    "3. I have no connection whatsoever to the Instagram account '@shadow_watch_blr', the phone number "
    "+91 91234 56789, or the UPI ID 'rohan.v88@okicici' (HDFC Account ending 4491)."
)
doc.save(os.path.join(OUT_DIR, "06_suspect_alibi_statement.docx"))

# 7. Bank / UPI Fraud Attempt Log (.csv)
upi_csv = """timestamp_ist,transaction_ref,requester_upi,linked_account_hint,target_victim,amount_inr,status,device_ip
2026-10-10 20:05:12,UPI6283910441,rohan.v88@okicici,HDFC-XXXX-4491,riya.sharma@okaxis,85000,COLLECT_REQUEST_DECLINED,49.207.212.88
2026-10-13 21:14:55,UPI6289941208,rohan.v88@okicici,HDFC-XXXX-4491,riya.sharma@okaxis,85000,COLLECT_REQUEST_EXPIRED,49.207.212.88
"""
with open(os.path.join(OUT_DIR, "07_upi_collect_request_log.csv"), "w", encoding="utf-8") as f:
    f.write(upi_csv)

print("Generated sample files:", sorted(os.listdir(OUT_DIR)))
