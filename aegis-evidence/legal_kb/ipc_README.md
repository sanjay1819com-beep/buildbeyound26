---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-retrieval
  - text-classification
tags:
  - legal
  - law
  - india
  - indian-penal-code
  - statute-identification
  - retrieval
pretty_name: Indian IPC Statute Identification
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files: data/*.parquet
---

# Indian IPC Statute Identification

Given the **facts of an Indian court case**, identify the relevant **Indian Penal Code (IPC) section**. Each example
pairs the factual narrative of a High Court judgment with the text of an IPC section that the judgment applies.

The task is framed as retrieval / statute identification: from a fact scenario, retrieve (or classify) the governing
statute. It is a useful benchmark and training signal for legal information retrieval, legal text classification, and
domain adaptation of text embedding models to Indian criminal law.

## Contents

| column | description |
|--------|-------------|
| `case_facts` | The factual narrative from an Indian High Court judgment. Explicit references to the governing section (e.g. "Section 302 IPC", "u/s 420") are **masked as `[STATUTE]`** so the example is a genuine facts→statute task and cannot be solved by string-matching the section number. |
| `ipc_section` | The relevant IPC section number (e.g. `302`, `304B`, `498A`). |
| `statute_text` | The full operative text of that IPC section — the offence definition and punishment (the "Whoever… shall be punished…" statutory language). |

- **40,001** examples · **~14,900** distinct cases · **320** distinct IPC sections.
- A case charged under multiple sections appears as multiple rows (one per relevant section).
- The section distribution reflects real criminal litigation — the most frequent sections are the common offences
  (murder §302, culpable homicide §304, hurt §323/§325, theft §379, cheating §420, cruelty §498A, rioting §147, …).

## How it was built

1. **Judgments**: factual narratives are drawn from Indian High Court judgments.
2. **Statute mapping**: each judgment's *own* citations to IPC sections are extracted (e.g. "under Section 302 IPC",
   "u/s 420") — a factual reference contained in the public record of the judgment. Those citations are then **masked**
   in `case_facts` so the model must infer the statute from the facts rather than read it off.
3. **Statute text**: the `statute_text` for each section is the corresponding IPC provision (offence + punishment).

## Sources & licensing

- **Court judgments** are Indian High Court decisions, redistributed here in reformatted, excerpted form under
  **CC-BY-4.0** — see NOTICE for attribution. Indian court judgments are themselves public records
  (Indian Copyright Act, 1957, §52(1)(q)).
- **IPC statute text** is the text of the Indian Penal Code, a public-domain enactment of the Government of India.
- **Curation** (citation extraction, masking, formatting) © 2026 Clause Logic Inc.

This dataset is released under **CC-BY-4.0**: free to use, including commercially, with attribution.

## Disclaimer

Provided "as is", without warranty of any kind. This dataset is for research and machine-learning purposes and does
**not** constitute legal advice.
