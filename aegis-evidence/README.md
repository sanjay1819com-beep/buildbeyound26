# AEGIS — AI-Powered Digital Evidence Intelligence Platform

**Problem Statement 3.2** · Indian criminal-law aware · built to be demonstrated live to judges.

A victim of harassment, fraud or stalking ends up with eight months of scattered
screenshots, a WhatsApp export, an `.eml`, a voicemail and a bank CSV — and no
way to turn that pile into something a court will accept. AEGIS ingests that
pile, seals it, correlates it across modalities, and outputs a cited narrative
together with the statutory certificate that makes it admissible.

It runs as a **live website** with a one-click sample case, so the whole pipeline
can be demonstrated in front of an audience.

```bash
cd /home/user/aegis-evidence && python3 server.py     # http://0.0.0.0:8080
```

---

## What it does

| Stage | Implementation |
|---|---|
| **Ingest & seal** | WORM vault, `chmod 0444`, SHA-256 + SHA-512 at arrival, original bytes never modified |
| **Parse** | `.txt`/chat, `.eml`, `.png`/`.jpg`, `.wav`, `.docx`, `.csv` — each parser emits one normalised contract |
| **Extract** | screen text from PNG chunks, entities, events, forensic audio transcript |
| **Deduplicate** | SHA-256 exact → 64-bit dHash perceptual → TF-IDF cosine; duplicates are **quarantined, never deleted** |
| **Correlate** | cross-modal alias resolution (burner phone ↔ anonymous handle ↔ ProtonMail ↔ UPI VPA) with citation-bearing inference edges |
| **Timeline** | dual-clock: *verified* time (SMTP relay, gateway log, carrier record) separated from *client-written* time (`Date:` header, EXIF, export text) |
| **Contradict** | five inconsistency classes, including version divergence between near-duplicates |
| **Charge** | tri-engine Indian statutory analysis → BNS / BNSS / BSA 2023 + IT Act 2000 |
| **Authenticate** | six edit-signature tests on screenshots, fused and calibrated against ground truth |
| **Certify** | printable BSA 2023 §63(4) Part A + Part B certificate with the hash pre-filled |
| **Share** | time-limited, revocable, role-scoped tokens with optional PII redaction |
| **Audit** | hash-chained ledger; every engine run is itself a block, and a live tamper test flips and restores an exhibit |

---

## The two v2 engines

### 1 · Image forensics — *is this screenshot fake or real?*

Classic photographic tests (ELA, noise residual, anti-aliasing drift, resampling)
**do not work on screenshots**, and we measured exactly how badly. Our first
implementation scored a fused AUC of **0.508** — a coin flip — and false-flagged
**5 of 5** genuine exhibits at 54–76%. A natively rendered UI screenshot
legitimately contains everything those tests hunt for: axis-aligned edges,
repeated identical bubbles, flat colour abutting crisp anti-aliased text. There
is no sensor noise and no lens to characterise. Published research says the same
(ELA is called "the single most abused technique in amateur forensics").

So v3 detects what an **editor** leaves behind:

| Test | Signature |
|---|---|
| `T1_white_mark_fill` | A solid rectangle whose colour is **near-but-not-equal to its own surround**. An eraser samples the neighbouring pixel and rounds, so it fills with *almost* the right colour. A genuine app header sits against the background — a long way away — so the guard rejects full-bleed chrome and requires a consistent surround on ≥3 sides. |
| `T2_foreign_patch` | A contiguous rectangle of blocks using colours that appear nowhere else in the document's own palette. |
| `T3_copy_move_clone` | Shift-invariant ORB keypoints voting on a consistent translation vector. |
| `T4_glyph_colour_drift` | A contiguous run of glyphs rendered in an off-palette colour — text re-typed by a different renderer. |
| `T5_jpeg_grid` | 8×8 DCT seam energy inside a **lossless** container. |
| `T6_container_metadata` | PNG chunk walk for `eXIf`/embedded JPEG; JPEG double-quantisation-table probe. |

Fusion is a **noisy-OR** over reliability-discounted scores, not a weighted mean:
independent signatures must be able to carry a verdict alone. Under a weighted
mean a single test at 49% with weight 0.26 contributed 13% and a real splice read
as "no manipulation" — which is precisely how v2 ended up useless.

**Measured, published, reproducible** (`python3 learn_forensics.py`, 72 ground-truth
images in `forensic_corpus/` plus the 3 real demo exhibits):

```
Genuine exhibits false-flagged      0 / 15      (genuine mean 0.0%, max 0.0%)
Fused AUC                           0.700       (v1 was 0.508)
eraser        9/12  (75.0%)         jpeg_splice  7/12  (58.3%)
paste         3/12  (25.0%)         clone        3/12  (25.0%)
overtype      0/12  ( 0.0%)  <-- published limitation, see below
```

**The honest limitation.** A same-font, near-same-colour overtype leaves no
chromatic trace and no pixel test detected it. That class is caught by the
**semantic** engine instead: the doctored copy contradicts the original because a
line is missing from it. In the live demo both engines independently converge on
the same forgery — `EV-008` reads 100% LIKELY MANIPULATED at pixel level
(localisation IoU **1.000** against the true edit box) and `INC-05`
VERSION_DIVERGENCE_CONTENT_LOSS at semantic level, naming the exact extortion
line that was removed.

No result here is a conclusion. A low score means *no edit signature was found*;
it does not prove authenticity. Allegations of manipulation require a
**BSA 2023 §63(4) Part B** expert opinion.

### 2 · Law of India — *which Act does this crime fall under?*

Three engines vote, because one cannot do the job:

- **A · precedent classifier** — `models/legal_ai.joblib`, trained on **40,001
  Indian High Court fact narratives** → IPC section (section references masked by
  the publisher, so the task cannot be string-matched). TF-IDF word 1-2g +
  `char_wb` 3-4g → LinearSVC → sigmoid calibration, 72 sections. Held-out:
  top-1 **10.4%**, top-5 **45.4%**, top-10 **66.0%**, calibrated top-1 **15.5%**.
- **B · deterministic rule engine** — 16 curated fact-pattern rules mapping
  straight to BNS sections, each backed by a **literal matched phrase** and the
  exhibit ID it came from.
- **C · statutory crosswalk** — 569 IPC→BNS, 190 IEA→BSA and CrPC→BNSS published
  mappings, plus 358 BNS / 531 BNSS / BSA section texts. Punishment clauses are
  extracted from the statute itself and are **sub-section aware** (`351(4)` gets
  the anonymous-intimidation bracket, not the base offence).

**Why three?** The precedent corpus has **zero** rows for stalking (IPC 354D),
voyeurism (354C), conspiracy (120B) and abetment (109), and one row for criminal
intimidation (503). A classifier cannot predict a class it has never seen, so a
single-engine design would **silently fail to charge the primary offence in a
harassment case**. Engine B carries those sections and every such charge is
flagged `PRECEDENT-MODEL BLIND SPOT` in the UI. On the live case, **5 of 10**
charges are blind-spot charges — including stalking itself.

Ranks that clear no bar are reported as **LEAD**, never as a charge: a calibrated
2–3% rank put "rioting" into a stalking-and-extortion fact pattern, and that is
labelled as noise rather than dressed up as a finding.

The engine also states what must **not** be cited — IT Act **§66A**, struck down
in *Shreya Singhal v. Union of India* (2015) 5 SCC 1, still registered in
practice because it survives in template FIRs; §79(3)(b), read down in the same
case; and the pre-2023 codes for offences on or after 1 July 2024.

> Decision-support aid. Not legal advice, not a charge sheet. Every citation
> resolves to published statutory text so a human can check it in one click.

---

## Live demo case — `CASE_BLR_2026_094`

Eight artifacts, one click (`Load sample case`, or press **L**):

| ID | File | Role |
|---|---|---|
| EV-001 | `01_whatsapp_chat_export.txt` | stalking + threat history |
| EV-002 | `02_threatening_email_raw.eml` | extortion demand, **backdated header** |
| EV-003 | `03_instagram_dm_screenshot.png` | victim's genuine screenshot → **0.0%** |
| EV-004 | `04_forwarded_duplicate_ig_screenshot.png` | near-duplicate, 95.3% dHash |
| EV-005 | `05_threatening_voicemail_oct13.wav` | audio threat |
| EV-006 | `06_suspect_alibi_statement.docx` | the alibi that collapses |
| EV-007 | `07_upi_collect_request_log.csv` | the financial anchor |
| EV-008 | `08_doctored_instagram_dm_screenshot.png` | **the accused's version → 100.0% LIKELY MANIPULATED** |

What the platform finds: **5 inconsistencies** (geo-contradiction, 6-day header
backdating, alibi falsification, identity-attribution conflict, version
divergence), **17 entities** across an alias chain joining 7 identifiers to one
natural person at 98%, **10 BNS provisions**, and a §63(4) certificate ready to
print.

`EV-008` is produced by `make_tampered_sample.py`, which erases the extortion
demand from `EV-003` the way an amateur would — sampling the bubble colour and
rounding — then asserts that the detector catches it *and* that the genuine
original stays clean. The script fails loudly if either is untrue.

---

## API

```
GET  /                              single-file UI, no CDN, no build step
GET  /api/state                     everything the UI renders
POST /api/load-sample               ingest the 8-artifact demo case
POST /api/reset                     back to pristine
POST /api/upload                    multipart: files[], uploader, device
POST /api/verify                    re-hash the vault + walk the ledger
POST /api/simulate-tamper           {"mode":"tamper"|"restore"}
POST /api/share                     {recipient, role, expiry_hours, redact_pii}
POST /api/revoke-share              {token_id}
POST /api/bsa-certificate           §63(4) Part A + Part B, hash pre-filled

POST /api/legal-analysis            tri-engine statutory analysis          (v2)
GET  /api/section/{act}/{section}   BNS | BNSS | BSA operative text        (v2)
POST /api/image-forensics[/{ev_id}] screenshot authenticity               (v2)
GET  /api/forensics-heatmap/{ev_id} annotated exhibit overlay             (v2)
GET  /api/forensics-calibration     the detector's measured performance   (v2)
GET  /api/samples · /api/samples/{filename} · /api/vault/{ev_id}
```

18/18 endpoint tests pass, including negative cases (`BNS/9999` → 404,
`image-forensics/EV-999` → 404).

---

## Layout

```
server.py                 aiohttp backend + EvidencePlatformEngine
static/index.html         single-file frontend (13 views, vanilla JS, inline SVG)
legal_engine.py           tri-engine statutory analysis (v2)
image_forensics.py        calibrated screenshot authenticity (v2)
learn_forensics.py        measures the detector, learns fusion weights
train_legal_v2.py         memory-safe legal-model trainer
make_forensic_corpus.py   builds the 72-image ground-truth corpus
make_tampered_sample.py   builds EV-008 and asserts it is caught
generate_samples.py       builds the other 7 demo artifacts
probe_features.py         feature-evaluation harness (measure before wiring)
calibrate_forensics.py    older calibration harness
train_legal_ai.py         first trainer — SUPERSEDED, OOM-kills on 2 GB; kept for the record
models/legal_ai.joblib            31.8 MB — vectors, classifier, crosswalk, statute text
models/legal_ai_metrics.json      held-out accuracy
models/forensic_calibration.json  measured per-operation sensitivity
models/forensic_fusion.json       learned per-test reliability λ
legal_kb/                 BNS/BNSS/BSA sections + IPC↔BNS, IEA↔BSA, CrPC↔BNSS maps
forensic_corpus/          12 genuine + 60 tampered PNGs with manifest.json
sample_evidence/          the 8 demo artifacts
vault/                    WORM store, {ev_id}__{filename}, chmod 0444
```

Runtime needs only `aiohttp`, `Pillow`, `python-docx`, `scikit-learn`, `scipy`,
`numpy`, `opencv-python` and `joblib`. **No** torch, transformers, tesseract or
network access at runtime — the model and the statutes are bundled.

---

## Sources

Hugging Face: `Hanno-Labs/indian-ipc-statute-identification` (CC-BY-4.0),
`GSMS-B/indian-legal-sections-bns-bnss-bsa-2023` (Apache-2.0),
`GSMS-B/ipc-bns-crpc-bnss-iea-bsa-respective-old-new-sections-mapping` (CC-BY-4.0).

NCRB *Crime in India 2024* (cybercrime +17.9%, 86,420 → 1,01,928; rate 6.2 → 7.3
per lakh; 72.6% fraud motive; >1.2 lakh pending investigation) · BOOM on FIR
conversion (1.9–2.6%, 2020–2023) · Internet Freedom Foundation 2023 (~68% of
victims never reported) · *Forensic Science International: Digital Investigation*
(2026) Indicator-of-Inconsistency framework · iPleaders / Lawsikho / Bar & Bench
on BSA §63 and the Schedule certificate · vendor documentation and forensic trade
press on Cellebrite Genesis / Guardian / Pathfinder, Magnet AXIOM and OpenText
EnCase · AIP Advances and arXiv on ELA and JPEG ghost analysis · Amped
Authenticate, FotoForensics and Forensically on image-authentication practice.

*Anvar P.V. v. P.K. Basheer* (2014) · *Arjun Panditrao Khotkal v. Kailash
Gorantyal* (2020) · *Shreya Singhal v. Union of India* (2015) 5 SCC 1.
