---
language:
- en
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: train
    path: bns_bnss_bsa_sections.json
- config_name: bns
  data_files:
  - split: train
    path: bns_sections.json
- config_name: bnss
  data_files:
  - split: train
    path: bnss_sections.json
- config_name: bsa
  data_files:
  - split: train
    path: bsa_sections.json
tags:
- legal
- indian-law
- criminal-law
- bns
- bnss
- bsa
- bharatiya-nyaya-sanhita
- bharatiya-nagarik-suraksha-sanhita
- bharatiya-sakshya-adhiniyam
- bharatiya-nyaya-sanhita-2023
- bharatiya-nagarik-suraksha-sanhita-2023
- bharatiya-sakshya-adhiniyam-2023
- indian-penal-code-replacement
- code-of-criminal-procedure-replacement
- indian-evidence-act-replacement
- indian-criminal-law
- indian-legal-nlp
- law
- rag
- retrieval-augmented-generation
- question-answering
- text-classification
- feature-extraction
- legal-ai
- legal-nlp
- structured-legal-data
- nlp
pretty_name: Indian Legal Sections — BNS, BNSS & BSA 2023 (Structured JSON)
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-classification
- feature-extraction
---

<div align="center">

![Dataset Banner](assets/banner.png)

# 🏛️ Indian Legal Sections — BNS · BNSS · BSA 2023

### The First Structured, Unified JSON Dataset of Modern Indian Criminal Law

![Sections](https://img.shields.io/badge/Total%20Sections-1%2C059-navy?style=for-the-badge)
![Acts](https://img.shields.io/badge/Acts%20Covered-3-orange?style=for-the-badge)
![Chapters](https://img.shields.io/badge/Total%20Chapters-71-darkgreen?style=for-the-badge)
![Format](https://img.shields.io/badge/Format-Structured%20JSON-blue?style=for-the-badge&logo=json)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)
![Language](https://img.shields.io/badge/Language-English-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Verified-Accurate-brightgreen?style=for-the-badge)

</div>

---

## 📖 Dataset Summary

This dataset contains **1,059 fully structured and verified sections** extracted, parsed, and unified from India's three landmark criminal justice reform acts passed in December 2023. These three acts together replaced the colonial-era Indian Penal Code (IPC, 1860), the Code of Criminal Procedure (CrPC, 1973), and the Indian Evidence Act (IEA, 1872). They received Presidential assent on **December 25, 2023** and came into force on **July 1, 2024** — representing the most sweeping overhaul of India's criminal justice system in over 150 years. They are named as 2023 acts because that is the year of their parliamentary passage and enactment.

> [!IMPORTANT]
> **Verified Accuracy:** This dataset contains exactly **1,059 unified sections** mapping the complete current framework of modern Indian criminal law across all three acts, structured in a consistent JSON schema designed for immediate use in RAG pipelines, fine-tuning workflows, and legal NLP research.

<div align="center">

| Act | Full Name | 🔄 Replaces | 📚 Chapters | 📄 Sections |
| :--- | :--- | :--- | :---: | :---: |
| **BNS 2023** | Bharatiya Nyaya Sanhita, 2023 | Indian Penal Code, 1860 | 20 | 358 |
| **BNSS 2023** | Bharatiya Nagarik Suraksha Sanhita, 2023 | Code of Criminal Procedure, 1973 | 39 | 531 |
| **BSA 2023** | Bharatiya Sakshya Adhiniyam, 2023 | Indian Evidence Act, 1872 | 12 | 170 |
| **Total** | | | **71** | **1,059** |

</div>

All three acts operate together in every criminal case in India. The **BNS** defines what constitutes a crime and prescribes the punishment for it. The **BNSS** governs the entire criminal procedure from FIR registration through investigation, bail, charge, trial, judgment, and execution of sentence. The **BSA** determines what evidence is admissible in court and how it must be proved. This dataset unifies all 1,059 sections from all three acts into a single consistent machine-readable JSON format.

---

## 🏛️ Background: Why These Acts Matter

India's criminal justice system was governed for over 160 years by three laws written under British colonial administration. The Indian Penal Code (1860) defined crimes, the Code of Criminal Procedure (1973) governed how those crimes were investigated and tried, and the Indian Evidence Act (1872) governed what could be presented as proof in court.

The BNS, BNSS, and BSA were passed by the Parliament of India in the Winter Session of December 2023. They modernize the criminal justice framework by introducing:

- ⚖️ Provisions for organised crime and terrorism as distinct offences
- 💻 Electronic records recognized as **primary evidence** (not secondary)
- ⏱️ Mandatory timelines for investigation and trial completion
- 🤝 Community service introduced as a form of punishment
- 📋 Zero FIR provisions allowing registration at any police station regardless of jurisdiction
- 🆔 Updated definitions of offences to reflect 21st-century realities

For AI and NLP researchers, legal technologists, and developers building Indian legal AI systems, these three acts are the foundational texts of modern Indian criminal law and the correct source for any system targeting the current legal framework.

---

## 🗂️ Dataset Structure

### 📂 Available Files

This repository contains both the combined dataset and individual files separated by Act. If you want the sections of a specific Act separately, you can go to the **Files** tab and download them individually:

* `bns_bnss_bsa_sections.json` (1,059 sections) — The complete combined dataset.
* `bns_sections.json` (358 sections) — Only the Bharatiya Nyaya Sanhita (BNS) sections.
* `bnss_sections.json` (531 sections) — Only the Bharatiya Nagarik Suraksha Sanhita (BNSS) sections.
* `bsa_sections.json` (170 sections) — Only the Bharatiya Sakshya Adhiniyam (BSA) sections.

*Note: The individual files use the exact same JSON schema and structure as the combined dataset described below.*

### JSON Schema

Each of the 1,059 entries is a JSON object with the following structure:

```json
{
  "chunk_id": "BNS_101",
  "text": "[Context: This section is from BNS 2023, CHAPTER VI OF OFFENCES AFFECTING THE HUMAN BODY. It covers Section 101: Murder.]\n\n101. Murder.— Except in the cases hereinafter excepted, culpable homicide is murder,—\n(a) if the act by which the death is caused is done with the intention of causing death; or\n(b) if the act by which the death is caused is done with the intention of causing such bodily injury as the offender knows to be likely to cause the death of the person to whom the harm is caused; or\n...",
  "section_number": "101",
  "section_title": "Murder",
  "chapter": "CHAPTER VI OF OFFENCES AFFECTING THE HUMAN BODY",
  "act": "BNS 2023",
  "source_label": "Section 101, BNS 2023"
}
```

### 📋 Field Descriptions

| Field | Type | Description |
| :--- | :---: | :--- |
| `chunk_id` | string | Unique identifier in format `ACT_SectionNumber`. BNS: `BNS_1`→`BNS_358`. BNSS: `BNSS_1`→`BNSS_531`. BSA: `BSA_1`→`BSA_170`. Prefixed per act, not globally sequential. |
| `text` | string | Complete self-contained section text. Includes a prepended context header identifying the act and chapter, followed by the full section body with all sub-clauses, illustrations, exceptions, explanations, and provisos. Primary content field. |
| `section_number` | string | Official section number as it appears in the act (e.g., `"101"`, `"47"`, `"3"`). |
| `section_title` | string | Official title of the section as it appears in the act (e.g., `"Murder"`, `"Arrest without warrant"`). |
| `chapter` | string | Full chapter heading under which this section falls within its respective act. |
| `act` | string | Source act. Exactly one of: `"BNS 2023"`, `"BNSS 2023"`, or `"BSA 2023"`. |
| `source_label` | string | Human-readable citation string (e.g., `"Section 101, BNS 2023"`). Designed for direct use as a citation tag in RAG outputs. |

### 🧱 Text Field Design

Every `text` field is intentionally structured with a context header prepended before the section content, making each chunk fully self-contained for RAG pipelines:

```
[Context: This section is from BNS 2023, CHAPTER VI OF OFFENCES AFFECTING THE HUMAN BODY.
It covers Section 101: Murder.]

101. Murder.— Except in the cases hereinafter excepted, culpable homicide is murder,—
(a) if the act by which the death is caused is done with the intention of causing death; or
(b) if the act by which the death is caused is done with the intention of causing such bodily
injury as the offender knows to be likely to cause the death of the person to whom the harm is caused; or
...
Exception 1.— Culpable homicide is not murder if the offender, whilst deprived of the power of
self-control by grave and sudden provocation...
Illustration.
(a) A shoots Z with the intention of killing him. Z dies in consequence. A commits murder.
```

---

## 📊 Dataset Statistics

<div align="center">

| 📏 Metric | ✅ Value |
| :--- | :---: |
| Total sections (chunks) | **1,059** |
| BNS 2023 sections | 358 |
| BNSS 2023 sections | 531 |
| BSA 2023 sections | 170 |
| BNS 2023 chapters | 20 |
| BNSS 2023 chapters | 39 |
| BSA 2023 chapters | 12 |
| Total chapters across all three acts | **71** |
| Total fields per entry | 7 |

</div>

---

## 📚 Coverage by Act

### ⚖️ BNS 2023 — Bharatiya Nyaya Sanhita (358 sections, 20 chapters)

The BNS is the substantive criminal law of India — it defines every offence and its punishment.

<details>
<summary><b>📐 Click to expand full chapter coverage — BNS 2023 (20 Chapters)</b></summary>

- **Chapter I** — Preliminary definitions and applicability of the act
- **Chapter II** — Punishments (imprisonment, fine, forfeiture of property, community service)
- **Chapter III** — General exceptions (child, unsound mind, intoxication, consent, necessity, private defence)
- **Chapter IV** — Abetment, criminal conspiracy, and attempt
- **Chapter V** — Offences against women and children (sexual assault, rape, trafficking, child exploitation)
- **Chapter VI** — Offences affecting the human body (murder, culpable homicide, grievous hurt, assault, kidnapping, abduction, wrongful confinement)
- **Chapter VII** — Offences against the state (waging war, seditious acts, terrorist acts)
- **Chapter VIII** — Offences relating to the Army, Navy and Air Force
- **Chapter IX** — Offences relating to elections
- **Chapter X** — Offences relating to coin, currency notes, bank notes and government stamps
- **Chapter XI** — Offences against public tranquility (unlawful assembly, rioting, affray)
- **Chapter XII** — Offences by or relating to public servants (bribery, corruption, misconduct)
- **Chapter XIII** — Contempts of the lawful authority of public servants
- **Chapter XIV** — False evidence and offences against public justice (perjury, fabricating evidence, obstructing justice)
- **Chapter XV** — Offences affecting public health, safety, convenience, decency and morals
- **Chapter XVI** — Offences relating to religion
- **Chapter XVII** — Offences against property (theft, extortion, robbery, dacoity, cheating, fraud, mischief, criminal trespass)
- **Chapter XVIII** — Offences relating to documents and property marks (forgery, counterfeiting)
- **Chapter XIX** — Criminal intimidation, insult, annoyance, defamation
- **Chapter XX** — Repeal and savings

</details>

---

### 🏛️ BNSS 2023 — Bharatiya Nagarik Suraksha Sanhita (531 sections, 39 chapters)

The BNSS is the procedural criminal law of India — it governs every step of the criminal justice process from the moment a crime is reported to the final execution of a sentence. With 531 sections across 39 chapters it is the largest of the three acts and the most comprehensive component of this dataset.

<details>
<summary><b>📐 Click to expand full chapter coverage — BNSS 2023 (39 Chapters)</b></summary>

- **Chapter I** — Preliminary definitions and interpretation
- **Chapter II** — Constitution of criminal courts and offices
- **Chapter III** — Powers of courts (Magistrate, Sessions, High Court)
- **Chapter IV** — Powers of superior officers of police and aid to magistrates
- **Chapter V** — Arrest of persons (with and without warrant, rights of arrested persons, medical examination)
- **Chapter VI** — Processes to compel appearance (summons, warrant, proclamation, attachment)
- **Chapter VII** — Processes to compel production of things (search warrants, general search)
- **Chapter VIII** — Reciprocal arrangements for assistance in certain matters
- **Chapter IX** — Security for keeping the peace and for good behaviour
- **Chapter X** — Order for maintenance of wives, children and parents
- **Chapter XI** — Maintenance of public order and tranquility
- **Chapter XII** — Preventive action of the police
- **Chapter XIII** — Information to the police and their powers to investigate (FIR, Zero FIR, investigation procedure, charge sheet)
- **Chapter XIV** — Jurisdiction of criminal courts in inquiries and trials
- **Chapter XV** — Conditions requisite for initiation of proceedings
- **Chapter XVI** — Complaints to magistrates
- **Chapter XVII** — Commencement of proceedings before magistrates
- **Chapter XVIII** — The charge (framing, joinder, alteration of charges)
- **Chapter XIX** — Trial before a Court of Session
- **Chapter XX** — Trial of warrant-cases by magistrates
- **Chapter XXI** — Trial of summons-cases by magistrates
- **Chapter XXII** — Summary trials
- **Chapter XXIII** — Plea bargaining
- **Chapter XXIV** — Attendance of persons confined or detained in prisons
- **Chapter XXV** — Evidence in inquiries and trials
- **Chapter XXVI** — General provisions as to inquiries and trials
- **Chapter XXVII** — Provisions as to accused persons of unsound mind
- **Chapter XXVIII** — Provisions as to offences affecting the administration of justice
- **Chapter XXIX** — The judgment
- **Chapter XXX** — Submission of death sentences for confirmation
- **Chapter XXXI** — Appeals
- **Chapter XXXII** — Reference and revision
- **Chapter XXXIII** — Transfer of criminal cases
- **Chapter XXXIV** — Execution, suspension, remission and commutation of sentences
- **Chapter XXXV** — Provisions as to bail and bonds
- **Chapter XXXVI** — Disposal of property
- **Chapter XXXVII** — Irregular proceedings
- **Chapter XXXVIII** — Limitation for taking cognizance of certain offences
- **Chapter XXXIX** — Miscellaneous provisions

</details>

---

### 📜 BSA 2023 — Bharatiya Sakshya Adhiniyam (170 sections, 12 chapters)

The BSA is the law of evidence in India — it governs what facts can be proved in court, how they must be proved, and what standards apply.

<details>
<summary><b>📐 Click to expand full chapter coverage — BSA 2023 (12 Chapters)</b></summary>

- **Chapter I** — Preliminary definitions and interpretation
- **Chapter II** — Relevancy of Facts (admissions, confessions, statements by persons who cannot be called as witnesses, expert opinion, character evidence)
- **Chapter III** — Facts Which Need Not Be Proved (judicially noticeable facts, admitted facts)
- **Chapter IV** — Of Oral Evidence (firsthand evidence requirement, electronic evidence as oral evidence)
- **Chapter V** — Of Documentary Evidence (primary and secondary evidence, electronic records as primary evidence, production of documents)
- **Chapter VI** — Of the Exclusion of Oral Evidence by Documentary Evidence
- **Chapter VII** — Of the Burden of Proof (general burden, special rules for specific offences, presumptions)
- **Chapter VIII** — Estoppel
- **Chapter IX** — Of Witnesses (competency, privilege, compellability)
- **Chapter X** — Of Examination of Witnesses (order of examination, cross-examination, leading questions, hostile witnesses, refreshing memory, impeaching credit)
- **Chapter XI** — Of Improper Admission and Rejection of Evidence
- **Chapter XII** — Repeal and Savings

</details>

---

## 🎯 Intended Uses

> [!NOTE]
> This dataset was built with three primary use cases in mind — RAG pipelines, fine-tuning, and multi-agent legal AI systems.

**🔍 RAG Systems for Indian Legal Q&A**
Each chunk is fully self-contained with its context header, making it directly usable with embedding models such as `BAAI/bge-base-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`, or `intfloat/multilingual-e5-base`, and vector databases such as ChromaDB, FAISS, Pinecone, LanceDB, or Weaviate. The `source_label` field is designed for direct use as a citation tag in generated outputs.

**🤖 Fine-tuning Legal Language Models**
The structured format with section titles, chapter context, act attribution, and full legal text including illustrations and exceptions is well-suited for generating synthetic instruction-response pairs for supervised fine-tuning (SFT) of small language models on Indian criminal law.

**🕸️ Legal AI Agents and Multi-Agent Systems**
The per-act structure enables building multi-agent systems where specialized agents handle different aspects of a legal query — a BNS agent for crime identification and punishment, a BNSS agent for procedural guidance, and a BSA agent for evidence requirements.

**📊 Legal NLP Research**
Section classification, legal named entity recognition, cross-act relationship mapping, legal question answering, and Indian criminal law language modeling.

**📚 Legal Education and Reference Tools**
Building tools that help law students, legal aid workers, journalists, and citizens understand the new criminal laws of India.

> [!WARNING]
> This dataset is intended for research, education, and development purposes only. It must not be used as a substitute for qualified legal advice. The dataset represents the text of the acts as extracted via automated PDF parsing and may contain minor formatting artifacts. Users should verify critical legal text against the official Government of India Gazette notifications.

---

## 🔧 Source and Parsing Methodology

**Source:** Official text of BNS 2023, Bharatiya Nagarik Suraksha Sanhita 2023, and Bharatiya Sakshya Adhiniyam 2023 as published by the Government of India.

**Enactment timeline:** All three acts were passed by the Parliament of India in the Winter Session of December 2023. They received Presidential assent on December 25, 2023. They are named as 2023 acts because that is the year of parliamentary passage and enactment. They came into force on **July 1, 2024**, replacing the IPC, CrPC, and IEA from that date.

**Parsing methodology:** PDF documents were parsed using `pdfplumber` for text extraction. Section boundaries were identified using rule-based chunking that detects section number patterns in the extracted text. Each section was extracted with its complete content including all sub-clauses, illustrations, exceptions, explanations, and provisos. A context header was prepended to each chunk for RAG compatibility. All 1,059 sections across the three acts were validated for completeness and correct attribution before final export to JSON.

---

## 📎 Citation

If you use this dataset in your research or projects, please cite:

```bibtex
@dataset{indian_legal_sections_bns_bnss_bsa_2023,
  title     = {Indian Legal Sections — BNS, BNSS and BSA 2023 (Structured JSON)},
  author    = {GSMS-B},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/GSMS-B/indian-legal-sections-bns-bnss-bsa-2023}
}
```

---

## 📄 License

This dataset is released under the **Apache 2.0 License**.

The underlying legal texts — Bharatiya Nyaya Sanhita 2023, Bharatiya Nagarik Suraksha Sanhita 2023, and Bharatiya Sakshya Adhiniyam 2023 — are official Government of India legislative documents published in the Official Gazette and are in the public domain under Indian copyright law as government works.

---

<div align="center">

![Dataset Bottom Banner](assets/banner_bottom.png)

Made with ❤️ for the Indian Legal AI Research Community

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-GSMS--B-yellow?style=for-the-badge)](https://huggingface.co/GSMS-B)

</div>
