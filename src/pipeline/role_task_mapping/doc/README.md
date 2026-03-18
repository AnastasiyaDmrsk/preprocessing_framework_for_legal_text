# Role-Task Mapping

This module creates a **role–task mapping** (“who does what?”) from

- **preprocessed legal text** (articles/paragraphs, lists, markers, etc.) and
- an **organigram in the XML format** (units/roles/subjects),

and returns a **structured task list** as XML, suitable for downstream BPMN generation and/or LLM-based semantic extraction.

---

## Quickstart

**Entry point:** `build_role_task_mapping(...)` in `src/pipeline/role_task_mapping/api.py`.

Input:
- `preprocessed_text: str` – preprocessing output (text with stable paragraph/sentence boundaries).
- `organigram: str` – organigram as an XML string (CPEE organisation schema).
- `api_key: str` – Gemini API key.
- `output_dir: Path` – output directory for artifacts.

Output:
- Return value: XML as `str`
- File: `${output_dir}/role_task_mapping.xml`

---

## Architecture

The implementation supports two extraction paths:

1. **Pure LLM** (`LLMTaskExtractor`)
   - The LLM extracts all tasks directly from the text.
   - Performers (actors) are derived from the organigram and passed into the prompt as the allowed selection.

2. **Hybrid (NLP + LLM)** (`HybridTaskExtractor`)
   - First, SpaCy + rules heuristically extract **task candidates**.
   - Then the LLM validates/corrects the candidates and adds missing tasks.
   - If too few candidates are found, it **falls back** to “Pure LLM”.

---

## Data flow

1. `build_role_task_mapping` selects the extractor:
   - `use_hybrid=True` → `HybridTaskExtractor`
   - otherwise → `LLMTaskExtractor`

2. `extract_and_save_tasks(...)` (`LLMTaskExtractor`)
   - builds a performer lookup table from the organigram (`_build_performer_lookup`)
   - extracts tasks (`_extract_tasks`)
   - optional: validates and completes tasks (`_validate_tasks` if `use_validator=True`)
   - assigns stable IDs (`t1`, `t2`, …)
   - resolves exception cross-references (`_resolve_cross_refs`)
   - serializes the result as XML (`_generate_tasks_xml`) and saves it to `output_path`

3. Result: `role_task_mapping.xml` containing tasks, performers, deontics, conditions/exceptions, and source references.

---

## Input formats

### 1) Preprocessed text

The module expects **preprocessed text** as a string. It works best if

- sentence and paragraph boundaries are stable (for source references),
- paragraph/list markers (e.g. `1`, `3(a)`) remain visible in the text,
- meta markers (e.g. `--PARALLEL GATEWAY--`) may appear.

On the NLP-candidate side, obvious markers/noise are skipped (`_SKIP_RE` in `const.py`).

### 2) Organigram XML

The LLM extractor primarily uses `subject` entries in the organigram:

```xml
<subject id="..." uid="...">
  <relation unit="Applicant" role="Applicant"/>
</subject>
```

From these `unit/role` pairs a lookup table is built, which is later used to robustly parse/normalize performer strings from the LLM output.

---

## Configuration (parameters)

Configured via `build_role_task_mapping(...)` in `api.py`:

- `model`: LLM model for extraction (default from `src/pipeline/organigram/const.py`)
- `use_hybrid`: enable the hybrid path (default: `True`)
- `spacy_model`: SpaCy model name (used in the hybrid path)
- `use_validator`: second LLM run as a strict validator (currently default `False` in `api.py`)
- `validator_model`: model for the validator

Additionally in hybrid mode (`HybridTaskExtractor`):
- `nlp_min_candidates`: minimum number of candidates; otherwise fallback to “Pure LLM” (default from `organigram.const`)

---

## Hybrid path: NLP candidate extraction

File: `nlp_task_extractor.py`

`NLPTaskCandidateExtractor` uses SpaCy for:

- sentence segmentation (`doc.sents`)
- dependencies (`Token.dep_`, `Token.head`, `Token.children`)
- lemmas (`Token.lemma_`)
- pattern matching (`Matcher`), e.g. for “is required to”

Heuristics:
- detection of deontic auxiliaries (`shall/may/should/must`) plus negation
- verb phrase extraction (recursive over subtree, excluding certain dependencies)
- splitting compound verbs (`conj` verbs)
- conditions from `advcl` clauses with markers like `if/when/unless/...`
- article/paragraph indexing via regex (e.g. `Article 11`, paragraph marker `3(a)`)

The output are `TaskCandidate` objects (see `models.py`).

---

## LLM path: prompting, parsing, performer resolution

Files:
- `const.py` – prompts + regex/heuristic constants
- `llm_task_extractor.py` – LLM calls + parser + XML writer

### Prompts

- `TASK_EXTRACTION_PROMPT`: pure LLM extraction
- `TASK_NLP_CANDIDATES_PROMPT`: LLM validates/cleans candidates + finds missing tasks
- `TASK_VALIDATION_PROMPT`: strict validator run (optional)

### Parsing the LLM response

The model must output blocks in the format:

- `---TASK---` … `---END---`
- fields: `label`, `performers`, `deontic_type`, `deontic_modality`, `conditions`, `exceptions`, `article`, `paragraph`

`_parse_task_blocks()` is intentionally tolerant (ignores unknown lines) and drops blocks without a `label`.

### Performer resolution

`_parse_performers()` accepts multiple notations:

- `unit: X, role: Y` (preferred)
- `X/Y`
- or free strings → mapped via lookup (unit/role from the organigram)

---

## Exceptions & cross-references

Exceptions can include `[CROSS_REF]` in the prompt output.

- without `[CROSS_REF]`: automatically references the same task (`ref = task.id`)
- with `[CROSS_REF]`: `ref` is heuristically set to the “best matching task” via simple word overlap

Implemented in `LLMTaskExtractor._resolve_cross_refs()`.

---

## Files

- `api.py`
  - Public API: `build_role_task_mapping(...)`
  - selects hybrid vs. LLM extractor and writes `role_task_mapping.xml`

- `role_task_mapper.py`
  - `HybridTaskExtractor`: inherits from `LLMTaskExtractor`
  - creates NLP candidates, validates them via LLM, fallback logic

- `nlp_task_extractor.py`
  - `NLPTaskCandidateExtractor`: rule-/SpaCy-based candidate generation

- `llm_task_extractor.py`
  - `LLMTaskExtractor`: Gemini calls, parser, validator, cross-refs, XML output

- `const.py`
  - prompts, regexes, and sets for deontics/conditions/noise filtering

- `models.py`
  - dataclasses (`Task`, `TaskCandidate`, `TaskPerformer`, `TaskException`)
