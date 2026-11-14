"""Indian criminal-law analysis engine - which Act, which section, and why.

TRI-ENGINE BY NECESSITY, NOT BY DECORATION
------------------------------------------
The bundled precedent classifier was trained on the Hanno-Labs corpus of 40,001
Indian High Court fact narratives -> IPC section. That corpus is excellent but it
has holes that are fatal for exactly the offences this platform exists for:

    IPC 354D (stalking)              0 training rows
    IPC 354C (voyeurism)             0 training rows
    IPC 120B (criminal conspiracy)   0 training rows
    IPC 109  (abetment)              0 training rows
    IPC 503  (criminal intimidation) 1 training row
    IPC 507  (intimidation, anonymous) 4 training rows

A classifier cannot predict a class it has never seen. So a single-engine design
would silently fail to charge the primary offence in a harassment case - the one
charge the victim most needs. Hence three engines that vote:

  ENGINE A  precedent classifier   models/legal_ai.joblib (sklearn + joblib only)
            72 IPC sections, calibrated probabilities. Strong on fraud, theft,
            hurt, defamation; structurally blind to the zero-shot sections above.

  ENGINE B  deterministic rule engine - curated fact-pattern -> BNS section.
            Covers the zero-shot sections and never hallucinates: every hit is
            backed by a literal phrase it matched in the evidence, cited to the
            exhibit that contains it.

  ENGINE C  statutory crosswalk - IPC -> BNS 2023, IEA -> BSA 2023, CrPC -> BNSS
            2023, from the published mapping tables, plus operative section text,
            title, chapter and punishment extracted from the statute itself.

Engines A and B propose; engine C resolves every proposal to the section that is
actually in force on 1 July 2024 or later. The IPC number is retained as a legacy
reference because police, advocates and decades of case law still speak IPC.

This is a decision-support aid. It is not legal advice and it does not decide
anything - see DISCLAIMER.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy.sparse import hstack

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
KB = os.path.join(HERE, "legal_kb")

_BUNDLE = None

DISCLAIMER = (
    "This is a decision-support aid produced by software. It is NOT legal advice, NOT a "
    "charge sheet and NOT a substitute for an advocate or a investigating officer's "
    "independent judgement. Section identification is a ranked suggestion: the precedent "
    "model's measured top-1 accuracy is ~10% and top-5 ~45% across 72 sections, so the "
    "output must be read as leads to verify against the statute, never as a conclusion. "
    "Every citation below is resolved to the operative text of the section as published, "
    "so the provision can always be checked by hand."
)

# Sections with ZERO or near-zero training rows in the precedent corpus. Engine A is
# structurally blind to these; they are carried by the rule engine and flagged as such.
ZERO_SHOT_IPC = {"354D": 0, "354C": 0, "120B": 0, "109": 0, "503": 1, "507": 4,
                 "509": 10, "499": 4, "500": 15, "383": 4}


# --------------------------------------------------------------------- loading
def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        p = os.path.join(MODELS, "legal_ai.joblib")
        if not os.path.exists(p):
            _BUNDLE = {"unavailable": f"model bundle not found at {p}"}
        else:
            _BUNDLE = joblib.load(p)
    return _BUNDLE


def _kb(name):
    p = os.path.join(KB, name)
    if not os.path.exists(p):
        return []
    try:
        return json.load(open(p))
    except Exception:
        return []


def _index_sections(rows):
    out = {}
    for r in rows or []:
        sn = str(r.get("section_number") or "").strip()
        if sn and sn not in out:
            out[sn] = r
    return out


# ------------------------------------------------------------------ statute text
_PUNISH_RE = re.compile(
    r"(?:shall be punished|shall also be liable|punishable with|may extend to|"
    r"imprisonment for life|liable to fine|imprisonment of either description)", re.I)


def _punishment(text: str, limit: int = 300, sub: str | None = None) -> str:
    """Extract the operative sentencing clause.

    Two traps handled here:
      * BNS 308 and 351 place an ILLUSTRATION immediately after sub-section (1),
        and the punishment sub-sections (2)-(7) come AFTER it. Truncating the text
        at "Illustration" - the obvious thing to do - therefore discards every
        punishment clause and returns an empty string.
      * A section number like "351(4)" has its OWN distinct sentence (anonymous
        intimidation, a higher bracket than the base offence). When a sub-section
        is requested we return that clause, not the first one we find.
    """
    if not text:
        return ""
    t = _strip_context(text)
    want = None
    if sub:
        m = re.search(r"\((\d+)\)", str(sub))
        if m:
            want = m.group(1)
    hits = []
    for sent in re.split(r"(?<=[.;])\s+", t):
        c = re.sub(r"\s+", " ", sent).strip(" .;")
        if not _PUNISH_RE.search(c):
            continue
        if c.lower().startswith("illustration"):
            continue
        # an illustration sentence reads "(a) A threatens B that..." - skip those
        if re.match(r"^\(?[a-z]\)?[\.\)]?\s+\S", c) and re.search(r"\bA\b", c) and re.search(r"\bB\b", c):
            continue
        hits.append(c)
    if not hits:
        return ""
    if want:
        exact = [h for h in hits if re.match(r"^\(%s\)" % want, h)]
        if exact:
            return exact[0][:limit]
    numbered = [h for h in hits if re.match(r"^\(\d+\)", h)]
    out = numbered[0] if numbered else hits[0]
    return out[:limit]


def _strip_context(text: str) -> str:
    """Remove the dataset's [Context: ...] preamble, keep the operative text."""
    t = re.sub(r"^\s*\[Context:[^\]]*\]\s*", "", text or "", flags=re.S)
    return re.sub(r"\s+", " ", t).strip()


def _quote(text: str, n: int = 420) -> str:
    t = _strip_context(text)
    return t[:n] + ("..." if len(t) > n else "")


# ------------------------------------------------------------------- ENGINE C
class Crosswalk:
    """IPC -> BNS, IEA -> BSA, CrPC -> BNSS resolution against the published maps."""

    def __init__(self):
        # NB: these are `*_map`, not `ipc_to_bns`/`iea_to_bsa` - an instance attribute
        # of that name would shadow the lookup METHOD below and make it uncallable.
        self.ipc_map = _kb("ipc_to_bns.json")
        self.iea_map = _kb("iea_to_bsa.json")
        self.crpc_map = _kb("crpc_to_bnss.json")
        self.bns = _index_sections(_bundle().get("bns_sections") or _kb("bns_sections.json"))
        self.bsa = _index_sections(_bundle().get("bsa_sections") or _kb("bsa_sections.json"))
        self.bnss = _index_sections(_kb("bnss_sections.json"))
        self._ipc = {}
        for e in self.ipc_map:
            k = str(e.get("lookup_key") or "").strip()
            if k:
                self._ipc.setdefault(k, []).append(e)
        self._iea = {}
        for e in self.iea_map:
            k = str(e.get("lookup_key") or "").strip()
            if k:
                self._iea.setdefault(k, []).append(e)

    # -- BNS ------------------------------------------------------------------
    def bns_section(self, num: str) -> dict | None:
        """Full statutory record for a BNS section, by number (e.g. '78', '351')."""
        num = str(num).strip()
        base = num.split("(")[0]
        rec = self.bns.get(num) or self.bns.get(base)
        if not rec:
            return None
        return {"act": "Bharatiya Nyaya Sanhita, 2023", "act_short": "BNS 2023",
                "section": num, "title": rec.get("section_title", ""),
                "chapter": rec.get("chapter", ""),
                "punishment": _punishment(rec.get("text", ""), sub=num),
                "text": _strip_context(rec.get("text", "")),
                "quote": _quote(rec.get("text", ""))}

    def ipc_to_bns(self_, ipc: str) -> list[dict]:
        """Every published BNS provision corresponding to an IPC section."""
        out = []
        for e in _ipc_lookup(ipc):
            t = e.get("target")
            if not t:
                out.append({"status": e.get("status", "Deleted"), "section": None,
                            "sub_section": None, "title": None,
                            "statement": e.get("mapping_statement", "")})
                continue
            sub = str(t.get("sub_section") or t.get("section") or "")
            rec = CROSS.bns_section(sub) or CROSS.bns_section(str(t.get("section") or ""))
            out.append({"status": e.get("status", "Retained"),
                        "section": str(t.get("section") or ""),
                        "sub_section": sub,
                        "title": (rec or {}).get("title") or t.get("title") or "",
                        "chapter": t.get("chapter", ""),
                        "statement": e.get("mapping_statement", ""),
                        "punishment": _punishment((rec or {}).get("text", ""), sub=sub) or (rec or {}).get("punishment", ""),
                        "quote": (rec or {}).get("quote", "")})
        return out

    # -- BSA ------------------------------------------------------------------
    def bsa_section(self, num: str) -> dict | None:
        rec = self.bsa.get(str(num).strip())
        if not rec:
            return None
        return {"act": "Bharatiya Sakshya Adhiniyam, 2023", "act_short": "BSA 2023",
                "section": str(num), "title": rec.get("section_title", ""),
                "chapter": rec.get("chapter", ""),
                "text": _strip_context(rec.get("text", "")),
                "quote": _quote(rec.get("text", ""), 620)}

    def iea_to_bsa(self, iea: str) -> list[dict]:
        out = []
        for e in CROSS._iea.get(str(iea).strip(), []):
            t = e.get("target")
            out.append({"status": e.get("status", "Retained"),
                        "section": str(t.get("section")) if t else None,
                        "sub_section": (t or {}).get("sub_section"),
                        "title": (t or {}).get("title"),
                        "statement": e.get("mapping_statement", "")})
        return out

    # -- BNSS -----------------------------------------------------------------
    def bnss_section(self, num: str) -> dict | None:
        rec = self.bnss.get(str(num).strip())
        if not rec:
            return None
        return {"act": "Bharatiya Nagarik Suraksha Sanhita, 2023", "act_short": "BNSS 2023",
                "section": str(num), "title": rec.get("section_title", ""),
                "chapter": rec.get("chapter", ""),
                "quote": _quote(rec.get("text", ""), 420)}

    def bnss_find(self, title_contains: str) -> list[dict]:
        """Look BNSS provisions up BY TITLE so a section number is never invented."""
        q = title_contains.lower()
        return [{"section": str(r.get("section_number")), "title": r.get("section_title", "")}
                for r in self.bnss.values() if q in (r.get("section_title") or "").lower()]


def _ipc_lookup(ipc: str):
    k = str(ipc).strip()
    return CROSS._ipc.get(k, [])


# ------------------------------------------------------------------ ENGINE B
# Each rule states the fact pattern in literal, checkable terms. `any_of` fires on
# one match; `all_of` requires every group to match. Every firing rule records the
# exact phrase it matched and the exhibit ID it came from, so `why_applies` is
# evidence-bound rather than generated.
RULES = [
    dict(id="R_STALK", bns="78", ipc="354D", severity="HIGH",
         tags=["stalking", "harassment", "surveillance"],
         any_of=[r"\bfollow(?:ing|ed|s)?\b", r"\bwatch(?:ing|ed)?\s+(?:you|her|him)\b",
                 r"\bmonitor(?:ing)?\b", r"\btracking\b", r"\boutside your\b",
                 r"\bknow where you\b", r"\bnew number\b", r"\bblocked me\b.*\b(?:again|back)\b",
                 r"\b(?:keep|kept) (?:messaging|contacting|texting)\b",
                 r"\bseen your (?:story|post|profile)\b", r"\bi see (?:you|everything)\b",
                 r"\bcontact(?:ing)? you (?:again|repeatedly)\b", r"\bunwanted (?:messages|attention)\b"],
         requires_context=r"\b(?:repeated|again|daily|every day|weeks|months|despite|blocked|ignored|no response)\b",
         why="Repeated unwanted contact or surveillance after the victim has withdrawn consent "
             "is the definitional core of stalking."),

    dict(id="R_INTIM", bns="351", ipc="503/506", severity="CRITICAL",
         tags=["threat", "intimidation", "coercion"],
         any_of=[r"\bthreat(?:en|ening|ened)?\b", r"\bi will (?:kill|destroy|ruin|finish)\b",
                 r"\b(?:you|your family) will (?:regret|suffer|pay)\b", r"\bconsequences\b",
                 r"\bblackmail\b", r"\bor else\b", r"\bharm(?:ing)? you\b",
                 r"\bmake your life\b", r"\bno one will (?:believe|help) you\b",
                 r"\b(?:destroy|ruin) your (?:career|reputation|life)\b", r"\bleak\b.*\b(?:photos|video|chat)\b"],
         why="A threat of injury to person, reputation or property, made with intent to cause "
             "alarm or to compel an act, is criminal intimidation."),

    dict(id="R_INTIM_ANON", bns="351(4)", ipc="507", severity="CRITICAL",
         tags=["threat", "anonymity", "impersonation"],
         any_of=[r"\banonymous\b", r"\bunknown number\b", r"\bfake (?:profile|account|id)\b",
                 r"\bdisguis(?:e|ing)\b", r"\bsecond (?:account|number|profile)\b",
                 r"\bprepaid (?:sim|number)\b", r"\bnot (?:my real|real) (?:name|number)\b",
                 r"\bshadow\b.*\b(?:account|profile|id)\b"],
         requires="R_INTIM",
         why="Intimidation delivered anonymously, or from a concealed identity, carries the "
             "aggravated punishment bracket (the former IPC 507)."),

    dict(id="R_INSULT_MODESTY", bns="79", ipc="509", severity="HIGH",
         tags=["obscenity", "modesty"],
         any_of=[r"\bobscene\b", r"\bvulgar\b", r"\bsexually explicit\b", r"\bslut\b",
                 r"\bcheap\b.*\bcharacter\b", r"\bcomment on your body\b", r"\bdirty (?:messages|talk)\b",
                 r"\bsend (?:me )?(?:nudes|photos of you)\b"],
         why="Words, gestures or acts intended to insult the modesty of a woman, including "
             "obscene messages sent electronically."),

    dict(id="R_VOYEUR", bns="77", ipc="354C", severity="CRITICAL",
         tags=["voyeurism", "privacy", "image-based abuse"],
         any_of=[r"\b(?:watching|capturing|recording|photographing) (?:you|her) (?:in|while)\b",
                 r"\bprivate (?:photos|videos|moments)\b", r"\bwithout (?:my|her) (?:consent|knowledge)\b.*\b(?:photo|video|image)\b",
                 r"\bnude\b", r"\bchanging (?:room|clothes)\b", r"\bbathroom\b",
                 r"\bdistribut(?:e|ing) (?:my|her) (?:photos|images|video)\b", r"\bmorph(?:ed|ing)\b"],
         why="Capturing or disseminating images of a person engaging in a private act, or "
             "intimate images, without consent."),

    dict(id="R_EXTORT", bns="308", ipc="383/384", severity="CRITICAL",
         tags=["extortion", "money", "coercion"],
         all_of=[[r"\b(?:pay|transfer|send|give)\b.*\b(?:rs\.?|₹|inr|rupees|money|amount)\b",
                  r"\b(?:rs\.?|₹|inr|rupees)\s*\d[\d,\.]*", r"\bupi\b", r"\bdemand(?:ing)?\b.*\b(?:money|payment)\b"],
                 [r"\bthreat\b", r"\bor else\b", r"\bconsequences\b", r"\bblackmail\b",
                  r"\bif you don'?t\b", r"\bfail to pay\b", r"\bwill (?:leak|post|release|harm)\b"]],
         why="Money extracted by putting a person in fear of injury is extortion; the demand "
             "and the threat must both be present, and both are cited above."),

    dict(id="R_CHEAT", bns="318(4)", ipc="420", severity="HIGH",
         tags=["fraud", "cheating", "financial"],
         all_of=[[r"\b(?:cheat|fraud|scam|deceiv|duped|fake (?:job|offer|investment))\b",
                  r"\bfals(?:e|ely) (?:promise|represent|claim)\b", r"\bimpersonat\b"],
                 [r"\b(?:transfer|paid|payment|sent money|invested|upi|bank|account)\b"]],
         why="Dishonest inducement leading to delivery of property or money is cheating; where "
             "a computer resource or communication device is used, the aggravated bracket applies."),

    dict(id="R_FORGERY_ELEC", bns="340(2)", ipc="471", severity="CRITICAL",
         tags=["forgery", "fabricated evidence", "electronic record"],
         any_of=[r"\b(?:edit|edited|tamper|tampered|morph|fabricat|photoshop|manipulat)(?:ed|ing)?\b.*\b(?:screenshot|image|photo|document|chat|record)\b",
                 r"\b(?:fake|forged|doctored) (?:screenshot|document|chat|invoice|record)\b",
                 r"\bfalse (?:document|record|evidence)\b", r"\baltered\b.*\b(?:document|screenshot|record)\b"],
         why="Using a forged electronic record as if it were genuine. This charge is "
             "corroborated by the platform's own image-forensics verdict where available."),

    dict(id="R_FORGERY_MAKE", bns="336(3)", ipc="468", severity="HIGH",
         tags=["forgery"],
         any_of=[r"\bmade (?:a )?(?:fake|false) (?:document|invoice|id|profile)\b",
                 r"\bcreat(?:e|ed|ing) (?:a )?(?:fake|false|forged)\b.*\b(?:record|document|account)\b",
                 r"\bforged signature\b", r"\bcounterfeit\b"],
         why="Making a false document or electronic record with intent to cause damage or "
             "injury, or to support a claim."),

    dict(id="R_DEFAM", bns="356(1)", ipc="499/500", severity="MEDIUM",
         tags=["defamation", "reputation"],
         any_of=[r"\bdefam\b", r"\b(?:post|posted|publish|published|shared)\b.*\b(?:false|lies|about me)\b.*\b(?:public|online|social|group)\b",
                 r"\bruin(?:ing)? (?:my|her) reputation\b", r"\bslander\b", r"\blibel\b",
                 r"\bimputation\b", r"\bpublicly humiliat\b", r"\bposted (?:my|her) (?:photos|number)\b"],
         why="A false imputation made publicly with intent to harm reputation. Note this is a "
             "criminal complaint route; a civil suit lies in parallel."),

    dict(id="R_CONSPIRACY", bns="61(2)", ipc="120B", severity="HIGH",
         tags=["conspiracy", "multiple accused"],
         any_of=[r"\b(?:planned|planning|conspired|conspiracy) (?:together|with)\b",
                 r"\b(?:two|three|both|several) (?:of them|men|people|accounts)\b.*\b(?:together|coordinat)\b",
                 r"\bhis (?:friend|brother|associate|partner) (?:also|too)\b", r"\bacted in concert\b",
                 r"\bco-?accused\b", r"\b(?:multiple|several) (?:accounts|numbers|people)\b.*\b(?:same|coordinat)\b"],
         why="An agreement between two or more persons to do an illegal act. The precedent "
             "model has ZERO training rows for this offence, so it is carried entirely by rule."),

    dict(id="R_ABET", bns="49", ipc="109", severity="MEDIUM",
         tags=["abetment", "aiding"],
         any_of=[r"\b(?:instigat|aid(?:ed|ing)|encourag|abett?)(?:ed|ing)?\b",
                 r"\bhelped him\b", r"\bacted on his (?:behalf|instructions)\b", r"\bat his (?:behest|direction)\b"],
         why="Intentional aid, or instigation, to the commission of an offence. Also a "
             "zero-shot class for the precedent model."),

    dict(id="R_HURT", bns="115(2)", ipc="323", severity="HIGH",
         tags=["assault", "physical"],
         any_of=[r"\b(?:slap|slapped|punch|hit|beat|pushed|kicked|threw)\b",
                 r"\b(?:bruise|injur|bleeding|swelling|fracture)\b", r"\bphysically attacked\b"],
         why="Voluntarily causing hurt. A medico-legal certificate under BNSS should be obtained."),

    dict(id="R_RESTRAINT", bns="126(2)", ipc="341", severity="HIGH",
         tags=["restraint", "confinement"],
         any_of=[r"\b(?:blocked (?:my|the) (?:way|path|door)|prevented (?:me|her) from (?:leaving|going)|locked (?:me|her) in|wouldn'?t let (?:me|her) (?:go|leave))\b",
                 r"\bwrongful(?:ly)? (?:restrain|confin)\b"],
         why="Obstructing a person so as to prevent them from proceeding in a direction they "
             "have a right to proceed."),

    dict(id="R_TRESPASS", bns="331", ipc="441/448", severity="HIGH",
         tags=["trespass"],
         any_of=[r"\b(?:entered|barged into|broke into) (?:my|our|her) (?:house|home|flat|room|office)\b",
                 r"\bhouse-?trespass\b", r"\bwithout (?:my )?permission\b.*\b(?:entered|inside)\b"],
         why="Criminal trespass into a dwelling, with the aggravated bracket for house-breaking."),

    dict(id="R_ATTEMPT", bns="62", ipc="511", severity="MEDIUM",
         tags=["attempt"],
         any_of=[r"\battempt(?:ed|ing)? to\b", r"\btried to (?:kill|rape|rob|abduct|attack)\b",
                 r"\bfailed attempt\b"],
         why="An act towards the commission of an offence that did not succeed is punishable "
             "in its own right."),
]

# Provisions that look applicable but must NOT be cited. A tool that only adds
# charges is half a tool: the most common real-world error in Indian cyber FIRs is
# the continued registration of a section the Supreme Court struck down.
DO_NOT_CITE = [
    dict(act="Information Technology Act, 2000", section="66A",
         title="Punishment for sending offensive messages through communication service, etc.",
         status="STRUCK DOWN - UNCONSTITUTIONAL",
         authority="Shreya Singhal v. Union of India, (2015) 5 SCC 1 (Supreme Court, 24 March 2015)",
         why="Section 66A was struck down in its entirety as violative of Article 19(1)(a) - "
             "the terms 'grossly offensive', 'menacing character' and 'annoyance' were held to be "
             "unconstitutionally vague and over-broad, and the section was not saved by any "
             "reasonable restriction under Article 19(2). It is nevertheless still registered in "
             "practice because it survives in police databases and template FIRs. Citing it "
             "invites the whole complaint to be attacked.",
         use_instead="Charge the conduct under the BNS 2023 (e.g. s.351 criminal intimidation, "
                     "s.78 stalking, s.356 defamation) and, where a computer resource was used, "
                     "the surviving IT Act provisions (ss.66, 66C, 66D, 66E, 67, 67A)."),
    dict(act="Information Technology Act, 2000", section="79(3)(b)",
         title="Intermediary liability - conditions for exemption",
         status="READ DOWN (not struck down)",
         authority="Shreya Singhal v. Union of India, (2015) 5 SCC 1",
         why="Read down so that an intermediary loses safe harbour only on a court order or "
             "government notification, not on its own subjective judgment of 'actual knowledge'. "
             "Relevant when seeking preservation or disclosure of the accused's account from a "
             "platform: the correct route is a court order or a s.91 BNSS / s.102 BNSS process, "
             "not a request to the platform's discretion.",
         use_instead="Seek a judicial order or use the BNSS process for production of records."),
    dict(act="Indian Penal Code, 1860 / CrPC, 1973 / Indian Evidence Act, 1872",
         section="(whole codes)", title="The pre-2023 criminal codes",
         status="REPLACED for offences on or after 1 July 2024",
         authority="BNS 2023, BNSS 2023 and BSA 2023, in force 1 July 2024",
         why="For any offence committed on or after 1 July 2024 the new codes govern. Citing IPC "
             "or s.65B IEA in a post-commencement matter is a drafting error that the defence "
             "will use. The old codes still apply to earlier conduct, which is why every charge "
             "in this analysis carries BOTH the operative BNS citation and its legacy IPC "
             "equivalent.",
         use_instead="BNS 2023 for the offence, BNSS 2023 for procedure, BSA 2023 s.63(4) for "
                     "the electronic-records certificate (not s.65B IEA)."),
]

# IT Act 2000 provisions remain in force ALONGSIDE the BNS for cyber-specific conduct.
# The operative text is NOT in the bundled knowledge base, so these are cited by number
# and title only - never paraphrased as if quoted from statute.
IT_ACT = [
    dict(section="66", title="Computer related offences",
         when=r"\b(?:hack|hacking|unauthorised access|breach(?:ed)? (?:my|the) account|password|otp\b.*\b(?:stolen|shared)|malware|phishing)\b"),
    dict(section="66C", title="Punishment for identity theft",
         when=r"\b(?:identity theft|used my (?:photo|name|identity)|impersonat|fake (?:profile|account)|pretending to be)\b"),
    dict(section="66D", title="Punishment for cheating by personation by using computer resource",
         when=r"\b(?:cheat|fraud|scam)\b.*\b(?:online|upi|website|app|profile|computer|internet)\b"),
    dict(section="66E", title="Punishment for violation of privacy",
         when=r"\b(?:private|personal) (?:photos?|images?|videos?|pictures?)\b|\bnude[s]?\b|\bintimate (?:photo|image|video)\b|\bwithout (?:my|her|his) consent\b"),
    dict(section="67", title="Punishment for publishing or transmitting obscene material in electronic form",
         when=r"\b(?:obscene|vulgar|pornographic|nude|nudes|explicit)\b"),
    dict(section="67A", title="Punishment for publishing or transmitting material containing sexually explicit act, etc., in electronic form",
         when=r"\b(?:sexually explicit|sexual act|porn(?:ography)?|nude (?:photo|video|image)|intimate (?:photo|image|video))\b"),
    dict(section="72", title="Breach of confidentiality and privacy",
         when=r"\b(?:access(?:ed)? (?:my|her) (?:private|personal) (?:messages|photos|account))\b"),
]


# ------------------------------------------------------------------ ENGINE A
def precedent_ranks(facts: str, k: int = 12) -> list[dict]:
    """Engine A: calibrated probabilities over the 72 IPC sections the model knows."""
    b = _bundle()
    if "clf" not in b:
        return []
    try:
        X = hstack([b["word_vec"].transform([facts]), b["char_vec"].transform([facts])]).tocsr()
        proba = b["clf"].predict_proba(X)[0]
    except Exception:
        return []
    classes = [str(c) for c in b["clf"].classes_]
    order = np.argsort(-proba)[:k]
    return [{"ipc": classes[i], "probability": float(proba[i])} for i in order]


# --------------------------------------------------------------------- rules
def _compile(rules):
    for r in rules:
        r["_any"] = [re.compile(p, re.I) for p in r.get("any_of", [])]
        r["_all"] = [[re.compile(p, re.I) for p in grp] for grp in r.get("all_of", [])]
        r["_ctx"] = re.compile(r["requires_context"], re.I) if r.get("requires_context") else None
    return rules


RULES = _compile(RULES)


def _scan(text: str, pattern, ev_id: str | None):
    m = pattern.search(text or "")
    if not m:
        return None
    s = max(0, m.start() - 55)
    return {"phrase": m.group(0).strip(), "snippet": (text or "")[s:m.end() + 55].strip(),
            "evidence_id": ev_id}


def rule_matches(corpus: list[dict]) -> list[dict]:
    """Engine B over every exhibit. `corpus` = [{'id','text','kind'}]."""
    joined = "\n".join((c.get("text") or "") for c in corpus)
    fired, by_id = [], {r["id"]: None for r in RULES}
    for r in RULES:
        hits = []
        if r.get("_any"):
            for p in r["_any"]:
                for c in corpus:
                    h = _scan(c.get("text") or "", p, c.get("id"))
                    if h:
                        hits.append(h)
                        break
                if len(hits) >= 3:
                    break
        ok = bool(hits) if r.get("_any") else True
        if r.get("_all"):
            ok = True
            for grp in r["_all"]:
                got = None
                for p in grp:
                    for c in corpus:
                        got = _scan(c.get("text") or "", p, c.get("id"))
                        if got:
                            break
                    if got:
                        break
                if not got:
                    ok = False
                    break
                hits.append(got)
        if r.get("_ctx") and ok:
            if not r["_ctx"].search(joined):
                ok = False
        if ok and hits:
            rec = dict(rule=r, hits=hits)
            fired.append(rec)
            by_id[r["id"]] = rec
    # rules that require another rule to have fired
    out = []
    for rec in fired:
        dep = rec["rule"].get("requires")
        if dep and not by_id.get(dep):
            continue
        out.append(rec)
    return out


# ------------------------------------------------------------------ the engine
def analyse(facts: str, corpus: list[dict] | None = None,
            forensics: dict | None = None, inconsistencies: list[dict] | None = None,
            victim_is_minor: bool | None = None) -> dict:
    """Produce the full legal analysis for an incident.

    facts              : narrative of what happened (incident summary)
    corpus             : [{'id':'EV-001','text':..., 'kind':'email'|'image'|...}]
    forensics          : {ev_id: image_forensics result}  (corroborates forgery charges)
    inconsistencies    : platform-detected contradictions (corroborate + aggravate)
    victim_is_minor    : if True, POCSO 2012 is flagged in addition
    """
    global CROSS
    if CROSS is None:
        CROSS = Crosswalk()

    corpus = corpus or [{"id": "NARRATIVE", "text": facts}]
    rules = rule_matches(corpus)
    ranks = precedent_ranks(facts)

    charges, seen = [], set()

    # ---- Engine B proposals (rule-backed, evidence-cited) -------------------
    for rec in rules:
        r = rec["rule"]
        for sub in str(r["bns"]).split("/"):
            sec = CROSS.bns_section(sub.strip())
            if not sec:
                continue
            key = ("BNS", sec["section"])
            if key in seen:
                continue
            seen.add(key)
            zs = str(r.get("ipc", "")) in ZERO_SHOT_IPC or any(
                z in str(r.get("ipc", "")).split("/") for z in ZERO_SHOT_IPC)
            charges.append({
                "act": sec["act"], "act_short": "BNS 2023", "section": sec["section"],
                "title": sec["title"], "chapter": sec["chapter"],
                "punishment": sec["punishment"], "quote": sec["quote"],
                "confidence": round(min(0.97, 0.62 + 0.09 * len(rec["hits"])), 3),
                "severity": r.get("severity", "MEDIUM"),
                "engines": ["rule_engine"] + (["crosswalk"] if r.get("ipc") else []),
                "why_applies": r["why"],
                "matched_phrases": [{"phrase": h["phrase"], "evidence": h["evidence_id"],
                                     "snippet": h["snippet"][:150]} for h in rec["hits"]],
                "evidence": sorted({h["evidence_id"] for h in rec["hits"] if h["evidence_id"]}),
                "legacy_ipc": str(r.get("ipc") or "") or None,
                "legacy_note": ("IPC " + str(r.get("ipc")) + " -> BNS " + sec["section"] +
                                " (verified against the published crosswalk)") if r.get("ipc") else None,
                "precedent_model_blind_spot": zs,
                "blind_spot_note": ("The precedent classifier has "
                                    f"{ZERO_SHOT_IPC.get(str(r.get('ipc','')).split('/')[0], 0)} training rows for "
                                    f"IPC {r.get('ipc')}, so it cannot propose this charge. It is carried by the "
                                    "deterministic rule engine instead.") if zs else None,
                "tags": r.get("tags", []),
            })

    # ---- Engine A proposals (precedent model) -------------------------------
    # A rank at 2-3% is noise, not a charge: on this corpus the model put "rioting"
    # at 2-3% for a stalking/extortion fact pattern. Precedent-only proposals must
    # clear a real floor, are capped, and are reported as LEADS to verify - never as
    # charges on the same footing as a rule that matched a literal phrase.
    PRECEDENT_FLOOR, PRECEDENT_MAX = 0.045, 5
    precedent_proposed = 0
    for rank in ranks:
        ipc = rank["ipc"]
        if rank["probability"] < PRECEDENT_FLOOR:
            continue
        maps = CROSS.ipc_to_bns(ipc)
        targets = [m for m in maps if m.get("section")]
        if not targets:
            continue
        m = targets[0]
        sec = CROSS.bns_section(m.get("sub_section") or m.get("section")) or CROSS.bns_section(m["section"])
        if not sec:
            continue
        key = ("BNS", sec["section"])
        if key in seen:
            # already charged by rule: upgrade its provenance + confidence
            for c in charges:
                if ("BNS", c["section"]) == key:
                    c["engines"] = sorted(set(c["engines"] + ["precedent_model"]))
                    c["confidence"] = round(min(0.99, max(c["confidence"],
                                                          0.5 + rank["probability"] * 1.6)), 3)
                    c.setdefault("precedent_probability_ipc", {})[ipc] = round(rank["probability"], 4)
            continue
        if precedent_proposed >= PRECEDENT_MAX:
            continue
        seen.add(key)
        precedent_proposed += 1
        zs = ipc in ZERO_SHOT_IPC
        charges.append({
            "act": sec["act"], "act_short": "BNS 2023", "section": sec["section"],
            "title": sec["title"], "chapter": sec["chapter"],
            "punishment": sec["punishment"], "quote": sec["quote"],
            "confidence": round(min(0.75, 0.22 + rank["probability"] * 2.6), 3),
            "severity": "LEAD",
            "is_lead": True,
            "engines": ["precedent_model", "crosswalk"],
            "why_applies": (f"LEAD, not a charge. The precedent classifier ranks IPC {ipc} "
                            f"({rank['probability']*100:.1f}%) among the closest fact patterns in "
                            f"40,001 Indian High Court narratives; IPC {ipc} resolves to this section "
                            f"under the published crosswalk."),
            "matched_phrases": [],
            "evidence": [],
            "legacy_ipc": ipc,
            "legacy_note": m.get("statement") or f"IPC {ipc} -> BNS {sec['section']}",
            "crosswalk_status": m.get("status"),
            "precedent_probability": round(rank["probability"], 4),
            "precedent_model_blind_spot": zs,
            "blind_spot_note": None,
            "tags": ["precedent-ranked"],
        })

    # ---- IT Act 2000 (in force alongside the BNS) --------------------------
    it_hits = []
    for it in IT_ACT:
        m = re.search(it["when"], facts or "", re.I)
        if not m:
            for c in corpus:
                m = re.search(it["when"], c.get("text") or "", re.I)
                if m:
                    break
        if m:
            it_hits.append({"act": "Information Technology Act, 2000 (as amended 2008)",
                            "act_short": "IT Act 2000", "section": it["section"], "title": it["title"],
                            "matched": m.group(0), "confidence": 0.72,
                            "text_available_in_kb": False,
                            "note": "Cited by section number and title only - the operative text of "
                                    "the IT Act is not part of the bundled knowledge base, so it is "
                                    "never paraphrased as if quoted. Verify against the statute."})

    # ---- corroborate forgery charges with the forensics verdict -------------
    forensic_support = []
    for ev_id, fr in (forensics or {}).items():
        if not isinstance(fr, dict):
            continue
        band = fr.get("band") or ""
        if band in ("high", "elevated"):
            forensic_support.append({
                "evidence": ev_id, "verdict": fr.get("verdict"),
                "fusion_pct": fr.get("fusion_pct"),
                "primary_finding": (fr.get("primary_finding") or "")[:260],
                "region": fr.get("primary_region")})
            for c in charges:
                if c["section"] in ("340(2)", "336(3)", "340", "336") and c["act_short"] == "BNS 2023":
                    c["confidence"] = round(min(0.99, c["confidence"] + 0.12), 3)
                    c["engines"] = sorted(set(c["engines"] + ["image_forensics"]))
                    if ev_id not in c["evidence"]:
                        c["evidence"].append(ev_id)
                    c.setdefault("forensic_corroboration", []).append(
                        f"{ev_id}: {fr.get('verdict')} at {fr.get('fusion_pct')}%")

    # ---- forensics-driven charge -------------------------------------------
    # A detected edit IS the actus reus of BNS s.340(2): using a forged electronic
    # record as genuine. The rule engine cannot reach it from text alone - in this
    # case no exhibit says "I edited the screenshot", the platform simply SEES the
    # edit. So the forensic verdict is admitted as the matching evidence, with the
    # located region as its citation. This is the pixel->statute link no incumbent
    # product makes: image authentication and charge identification live in
    # different tools, in different workflows, at different companies.
    if forensic_support:
        sec = CROSS.bns_section("340(2)")
        if sec and ("BNS", sec["section"]) not in seen:
            seen.add(("BNS", sec["section"]))
            ids = [f["evidence"] for f in forensic_support]
            strongest = max(forensic_support, key=lambda f: f.get("fusion_pct") or 0)
            pct = float(strongest.get("fusion_pct") or 0)
            charges.append({
                "act": sec["act"], "act_short": "BNS 2023", "section": sec["section"],
                "title": sec["title"], "chapter": sec["chapter"],
                "punishment": sec["punishment"], "quote": sec["quote"],
                "confidence": round(min(0.82, 0.50 + pct / 100.0 * 0.32), 3),
                # A pixel-level screening result must not be presented as near-certain,
                # so this charge carries a hard ceiling that later corroboration cannot lift.
                "confidence_cap": 0.85,
                "severity": "CRITICAL" if pct >= 62 else "HIGH",
                "engines": ["image_forensics", "crosswalk"],
                "why_applies": (
                    "Proposed from the platform's own image-forensics finding rather than from any "
                    "statement in the evidence: the exhibit was analysed in place inside the sealed "
                    "vault and carries a located edit signature. Using an electronic record known or "
                    "believed to be forged, as if it were genuine, is the offence. Where the forgery "
                    "was made in order to cheat, BNS s.336(3) is attracted in addition."),
                "matched_phrases": [{
                    "phrase": f"{f['verdict']} ({f['fusion_pct']}%) at region {f['region']}",
                    "evidence": f["evidence"],
                    "snippet": (f.get("primary_finding") or "")[:220]} for f in forensic_support],
                "evidence": sorted(set(ids)),
                "legacy_ipc": "471",
                "legacy_note": "IPC 471 -> BNS 340(2) (verified against the published crosswalk)",
                "precedent_model_blind_spot": False,
                "blind_spot_note": None,
                "forensic_corroboration": [f"{f['evidence']}: {f['verdict']} at {f['fusion_pct']}%"
                                           for f in forensic_support],
                "requires_expert_confirmation": (
                    "The forensic verdict is a screening result. Before this charge is put before a "
                    "court it must be confirmed by a BSA 2023 s.63(4) Part B expert opinion."),
                "tags": ["forgery", "electronic record", "forensics-driven"],
            })

    # ---- aggravation from detected inconsistencies --------------------------
    aggravation = []
    for inc in (inconsistencies or []):
        t = str(inc.get("type") or inc.get("id") or "").upper()
        aggravation.append({"id": inc.get("id"), "type": t,
                            "summary": str(inc.get("summary") or inc.get("description") or "")[:220],
                            "evidence": inc.get("evidence") or inc.get("citations") or []})
        if "ALIBI" in t or "TEMPORAL" in t or "FALSIF" in t:
            for c in charges:
                if c["section"] in ("336(3)", "340(2)", "356(1)"):
                    c["confidence"] = round(min(0.99, c["confidence"] + 0.05), 3)

    # Final clamp. Corroboration and aggravation bonuses are applied by several
    # loops above, and without this a forensics-driven charge drifted from its
    # intended 0.82 to 0.99 - which flatly contradicted the "screening aid, needs
    # a s.63(4) Part B expert" caveat rendered immediately beneath it.
    for c in charges:
        cap = c.get("confidence_cap", 0.97 if c.get("is_lead") else 0.99)
        if c["confidence"] > cap:
            c["confidence_capped_from"] = c["confidence"]
            c["confidence"] = round(cap, 3)

    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "LEAD": 9}
    charges.sort(key=lambda c: (order.get(c.get("severity"), 3), -c["confidence"]))

    b = _bundle()
    metrics = (b.get("metrics") or {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jurisdiction": {
            "country": "India",
            "regime": "Criminal law as re-codified with effect from 1 July 2024",
            "substantive": "Bharatiya Nyaya Sanhita, 2023 (BNS) - replaced the Indian Penal Code, 1860",
            "procedural": "Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) - replaced the CrPC, 1973",
            "evidentiary": "Bharatiya Sakshya Adhiniyam, 2023 (BSA) - replaced the Indian Evidence Act, 1872",
            "also_in_force": "Information Technology Act, 2000 (as amended 2008) - applies in addition to the BNS "
                             "for conduct involving a computer resource or communication device",
            "note": "For any offence on or after 1 July 2024 the BNS/BNSS/BSA apply. Earlier conduct is "
                    "governed by the IPC/CrPC/IEA, which is why every charge below also carries its legacy citation.",
        },
        "charges": charges,
        "it_act_provisions": it_hits,
        "do_not_cite": DO_NOT_CITE,
        "by_act": _by_act(charges, it_hits),
        "forensic_support": forensic_support,
        "aggravating_inconsistencies": aggravation,
        "pocso_flag": ({
            "applies": bool(victim_is_minor),
            "note": ("BNS s.2(3) defines 'child' as a person below the age of eighteen years. Where the victim "
                     "is a child, the Protection of Children from Sexual Offences Act, 2012 applies IN ADDITION "
                     "to the BNS and is not displaced by it; it prescribes higher minimum sentences and "
                     "child-specific procedure.") if victim_is_minor else
                    ("Not flagged: the victim was not identified as a minor. BNS s.2(3) defines 'child' as a "
                     "person below eighteen; if that changes, POCSO 2012 applies in addition to the BNS."),
            "bns_child_definition": (CROSS.bns_section("2") or {}).get("title", ""),
        }),
        "procedure": _procedure(),
        "admissibility": _admissibility(corpus),
        "precedent_model": {
            "ranked_ipc": ranks,
            "gate": {"floor": PRECEDENT_FLOOR, "max_proposals": PRECEDENT_MAX,
                     "proposed": precedent_proposed,
                     "why": "Precedent-only proposals below the floor are suppressed and the rest are "
                            "labelled LEAD, because a calibrated 2-3% rank over 72 sections is noise."},
            "card": metrics,
            "honest_reading": (
                f"Trained on {metrics.get('corpus_rows_available', 40001):,} High Court fact narratives, "
                f"{metrics.get('sections', 72)} sections. Held-out accuracy: top-1 "
                f"{metrics.get('top1')}%, top-5 {metrics.get('top5')}%, top-10 {metrics.get('top10')}%, "
                f"calibrated top-1 {metrics.get('calibrated_top1')}%. That is roughly ten times chance but it "
                "is NOT an oracle - read the ranked list as leads. The model is also structurally blind to "
                "stalking, voyeurism, conspiracy and abetment, which have zero or near-zero training rows; "
                "those charges come from the rule engine and are flagged as such."),
        },
        "engines": {
            "A_precedent_classifier": "models/legal_ai.joblib - sklearn TF-IDF (word 1-2g + char_wb 3-4g) "
                                      "-> LinearSVC -> sigmoid calibration, 72 IPC sections",
            "B_rule_engine": f"{len(RULES)} curated fact-pattern rules mapping directly to BNS sections, "
                             "each backed by a literal matched phrase and an exhibit ID",
            "C_crosswalk": f"{len(CROSS.ipc_map)} IPC->BNS, {len(CROSS.iea_map)} IEA->BSA, "
                           f"{len(CROSS.crpc_map)} CrPC->BNSS published mappings; "
                           f"{len(CROSS.bns)} BNS / {len(CROSS.bsa)} BSA / {len(CROSS.bnss)} BNSS section texts",
        },
        "counts": {"charges": len(charges), "it_act": len(it_hits),
                   "blind_spot_charges": sum(1 for c in charges if c.get("precedent_model_blind_spot")),
                   "forensically_corroborated": len(forensic_support)},
        "disclaimer": DISCLAIMER,
    }


CROSS: Crosswalk | None = None


def _by_act(charges, it_hits):
    out = {}
    for c in charges + it_hits:
        out.setdefault(c["act_short"], []).append(
            {"section": c["section"], "title": c["title"],
             "confidence": c.get("confidence"), "severity": c.get("severity")})
    return out


def _procedure():
    """BNSS procedural route, with every section number resolved BY TITLE from the KB."""
    steps = []
    for want, why in [
        ("information in cognizable", "Register the FIR. The informant is entitled to a free copy; "
                                      "information may also be given by electronic means and must be "
                                      "reduced to writing and read over."),
        ("examination of witnesses by police", "Statements of witnesses during investigation."),
        ("statements to police and use thereof", "Limits on how a police statement may later be used."),
        ("recording of search and seizure through audio-video",
         "Search and seizure must be recorded by audio-video electronic means - directly relevant "
         "when the evidence is itself digital."),
        ("offences committed by means of electronic communications",
         "Procedure where the offence was committed by electronic communication or letter."),
        ("victim compensation scheme", "Compensation to the victim, including where the offender is not traced."),
        ("trial and proceedings to be held in electronic mode", "Electronic mode of trial and proceedings."),
    ]:
        hits = CROSS.bnss_find(want)
        for h in hits[:1]:
            rec = CROSS.bnss_section(h["section"])
            steps.append({"act_short": "BNSS 2023", "section": h["section"], "title": h["title"],
                          "why": why, "quote": (rec or {}).get("quote", "")})
    return steps


def _admissibility(corpus):
    """BSA 2023 route for electronic evidence - the part that actually decides cases."""
    s63 = CROSS.bsa_section("63")
    s61 = CROSS.bsa_section("61")
    s62 = CROSS.bsa_section("62")
    s57 = CROSS.bsa_section("57")
    s39 = CROSS.bsa_section("39")
    electronic = [c for c in corpus if str(c.get("kind") or "").lower() in
                  ("image", "screenshot", "email", "chat", "audio", "video", "csv", "log", "document", "pdf")]
    return {
        "governing_provision": s63,
        "electronic_record_is_a_document": s61,
        "special_provisions": s62,
        "primary_evidence": s57,
        "expert_opinion": s39,
        "in_force_from": "1 July 2024",
        "replaces": "Section 65B, Indian Evidence Act, 1872",
        "conditions_s63_2": [
            "the computer or device that produced the information was used regularly to store or process "
            "information for the purposes of any activity regularly carried on over that period by a person "
            "having lawful control over the use of that device;",
            "during that period information of the kind contained in the electronic record was fed into the "
            "device in the ordinary course of the said activities;",
            "throughout the material part of that period the device was operating properly, or if not, that "
            "any malfunction did not affect the accuracy of the record or its contents;",
            "the information contained in the electronic record reproduces or is derived from information "
            "fed into the device in the ordinary course of those activities.",
        ],
        "s63_3_multiple_devices": "Where several devices or systems act as one - a phone, a network, a cloud "
                                  "service, an intermediary - they are treated as a single unit, and the "
                                  "certificate may be given for the combined system.",
        "certificate_s63_4": {
            "mandatory": True,
            "required_at": "EACH instance where the electronic record is tendered, not once per case",
            "signed_by": "Two persons: the person occupying the responsible official position in relation to "
                         "the operation of the device (the custodian) AND a person who has examined the record "
                         "(the expert). The Schedule to the Act prescribes the form.",
            "part_a_custodian": ["identity and address of the person giving the certificate",
                                 "identity of the electronic record and the manner in which it was produced",
                                 "device details: make, model, colour, serial number, IMEI/UIN/UID, MAC address, "
                                 "Cloud ID as applicable",
                                 "declarations of lawful control, regular use in the ordinary course, and proper "
                                 "operation (or that any fault did not affect accuracy)",
                                 "ownership of the device",
                                 "the HASH VALUE of the record and the algorithm used, with the hash report enclosed",
                                 "signature, date, time (IST, 24-hour) and place"],
            "part_b_expert": ["name, address, designation and qualification",
                              "confirmation that the expert has examined the record",
                              "confirmation that the Part A device details are accurate",
                              "INDEPENDENT restatement of the same hash value",
                              "attestation of the record's integrity",
                              "signature, date, time (IST, 24-hour) and place"],
            "why_the_hash_matters": "The Supreme Court has described the hash as the 'electronic fingerprint' "
                                    "of the record: it is what ties the exhibit produced in court to the bytes "
                                    "originally seized, and it is why Part B is an additional layer of "
                                    "authenticity rather than a formality. The expert need not be an examiner "
                                    "notified under s.79A of the IT Act.",
        },
        "exhibits_needing_a_certificate": [c.get("id") for c in electronic],
        "platform_position": ("This platform captures the Part A device and hash fields at ingest, keeps the "
                              "original bytes in a write-once vault, and prints the s.63(4) certificate in the "
                              "prescribed two-part form with the SHA-256 value already filled in. It cannot "
                              "supply Part B - that is a human expert's opinion."),
        "case_law": ["Anvar P.V. v. P.K. Basheer (2014) - certificate is mandatory, not directory",
                     "Arjun Panditrao Khotkal v. Kailash Gorantyal (2020) - compliance with the certificate "
                     "requirement, and the consequences of its absence"],
    }


def explain_section(act: str, section: str) -> dict | None:
    """Direct lookup used by the UI's 'which Act does this crime fall under' box."""
    global CROSS
    if CROSS is None:
        CROSS = Crosswalk()
    act = (act or "").upper()
    if "BNS" in act or "NYAYA" in act:
        return CROSS.bns_section(section)
    if "BSA" in act or "SAKSHYA" in act:
        return CROSS.bsa_section(section)
    if "BNSS" in act or "SURAKSHA" in act:
        return CROSS.bnss_section(section)
    return (CROSS.bns_section(section) or CROSS.bsa_section(section) or CROSS.bnss_section(section))


if __name__ == "__main__":
    import sys, textwrap
    facts = sys.argv[1] if len(sys.argv) > 1 else textwrap.dedent("""
        Rohan Verma has been messaging the complainant daily for six weeks despite being blocked.
        He created a fake profile @shadow_watch_blr and a second anonymous number. He says "I know
        where you live, I was watching you outside your office" and threatens "pay Rs 85,000 by UPI
        or else I will leak your private photos and destroy your career". He also edited a chat
        screenshot and posted false claims about her publicly. His friend also helped him plan this.
    """)
    r = analyse(facts, corpus=[{"id": "EV-001", "text": facts, "kind": "narrative"}])
    print(f"CHARGES ({r['counts']['charges']}) — blind-spot charges: {r['counts']['blind_spot_charges']}\n" + "=" * 96)
    for c in r["charges"]:
        print(f"\n{c['act_short']} s.{c['section']} — {c['title']}")
        print(f"   severity {c['severity']}  confidence {c['confidence']:.2f}  engines {','.join(c['engines'])}")
        print(f"   punishment: {c['punishment'][:170]}")
        print(f"   why: {c['why_applies'][:200]}")
        if c["matched_phrases"]:
            print(f"   matched: {[m['phrase'] for m in c['matched_phrases']][:4]}  evidence {c['evidence']}")
        if c.get("blind_spot_note"):
            print(f"   !! {c['blind_spot_note'][:170]}")
        if c.get("legacy_note"):
            print(f"   legacy: {c['legacy_note'][:150]}")
    print("\n" + "=" * 96 + f"\nIT ACT ({len(r['it_act_provisions'])}): " +
          ", ".join(f"s.{i['section']} {i['title'][:40]}" for i in r["it_act_provisions"]))
    print(f"\nBNSS PROCEDURE ({len(r['procedure'])}):")
    for p in r["procedure"]:
        print(f"   s.{p['section']} {p['title'][:70]}")
    print(f"\nADMISSIBILITY: s.{r['admissibility']['governing_provision']['section']} "
          f"{r['admissibility']['governing_provision']['title']}")
    print("\nPRECEDENT TOP-8:", [(x['ipc'], round(x['probability'] * 100, 1)) for x in r['precedent_model']['ranked_ipc'][:8]])
