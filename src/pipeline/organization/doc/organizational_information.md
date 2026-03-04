# Organizational Information Extraction & Organigram Generation

This repository implements an **organizational actor extraction** pipeline that turns **preprocessed legal text** into a **CPEE-compatible `organigram.xml`**. The design supports two modes:

- **Pure LLM extraction**: the LLM extracts actors and their hierarchies directly.
- **Hybrid extraction (recommended)**: NLP heuristics extract candidate actors first, then the LLM validates/corrects them and infers hierarchies.

The goal is to robustly extract:

- **Actors** as either **UNIT** (institutions/bodies) and/or **ROLE** (functions/positions)
- **Hierarchies**: UNIT–UNIT, ROLE–ROLE, and ROLE–UNIT relations
- A structured XML output (`organigram.xml`) suitable for downstream BPMN/CPEE tooling.

---

## High-level flow

1. **Input**: `preprocessed_text` (a string coming from the preprocessing step).
2. **Actor extraction**:
   - *Pure LLM*: LLM extracts `ACTOR_NAME | TYPE` lines.
   - *Hybrid*: NLP pipeline proposes candidates → LLM validates/corrects + adds missing actors.
3. **Hierarchy extraction** (LLM): infer parent-child relations between UNIT/ROLE and bind roles to units.
4. **Generate dummy subjects** (for CPEE `subjects` section).
5. **Serialize** to `organigram.xml`.

---

## Files and responsibilities

### `src/pipeline/organization/api.py`
Public entry point for the organigram step.

- Provides `build_organigram_xml(...)`.
- Selects extractor implementation:
  - `HybridOrganizationalExtractor` (NLP candidates + LLM validation) when `use_hybrid=True`.
  - `LLMOrganizationalExtractor` (pure LLM) when `use_hybrid=False`.
- Saves the resulting `organigram.xml` to the chosen `output_dir`.

Key function:
- `build_organigram_xml(preprocessed_text, api_key, output_dir, model, use_hybrid, spacy_model, use_gliner)`

---

### `src/pipeline/organization/organization.py`
Hybrid orchestrator that combines NLP candidate extraction with LLM validation.

- Extends `LLMOrganizationalExtractor` and overrides `_extract_actors`.
- Runs NLP candidate extraction first via `NLPActorCandidateExtractor`.
- Uses a configurable threshold (`DEFAULT_NLP_MIN_CANDIDATES`) to decide:
  - If NLP found too few candidates → fallback to pure-LLM extraction.
  - Otherwise → validate candidates with the LLM (`_llm_validate_candidates`).

Why this exists:
- Makes extraction more stable and interpretable.
- Improves recall on entities that spaCy can find reliably.
- Reduces LLM hallucinations by anchoring on explicit candidates.

---

### `src/pipeline/organization/nlp_extractor.py`
Multi-layer NLP pipeline to extract actor candidates from text.

Core ideas:
- Returns a list of `(name, type)` tuples where `type ∈ {"UNIT", "ROLE"}`.
- Allows the same surface name to appear as both UNIT and ROLE.
- Uses several complementary layers:

Layers:
1. **spaCy NER**: ORG/GPE/NORP → UNIT candidates.
2. **Dependency parsing + action-verb heuristic**: subjects/agents of process verbs → ROLE.
3. **PhraseMatcher**: matches a curated list of EU institutions → UNIT.
4. **Frequent noun-phrase subjects**: frequency-weighted candidates.
5. **Optional GLiNER** (zero-shot NER) for higher recall.

Post-processing:
- Specificity filter removes generic single lowercase nouns.
- Plural normalization via lemmatization (e.g., “Member States” → “Member State”).

Configurable aspects:
- `spacy_model` (e.g., `en_core_web_sm`)
- `use_gliner`, `gliner_model`, `gliner_threshold`
- `freq_min_count`

---

### `src/pipeline/organization/llm_extractor.py`
Pure-LLM extraction and structuring logic.

Responsibilities:
- Calls the LLM (via `google.genai`) to:
  1. Extract actors (`_extract_actors`).
  2. Infer hierarchies and UNIT bindings (`_classify_and_structure_entities`).
- Produces structured objects (`OrganizationalEntity`, `Subject`).
- Generates dummy `subjects` to satisfy the CPEE organigram format (useful for tooling even when real people aren’t known).

Notable details:
- `_parse_actor_response()` strictly parses `ACTOR_NAME | TYPE` lines.
- Hierarchy output is parsed from `CHILD | PARENT | REL_TYPE` lines.
- Ensures every ROLE has at least one parent UNIT; adds `External` if needed.

---

### `src/pipeline/organization/const.py`
All constants and prompt templates.

Contains:
- Linguistic heuristics and filters:
  - `ACTOR_DEPS`, `PROCESS_ACTION_VERBS`
  - suffix heuristics (`UNIT_SUFFIXES`, `ROLE_SUFFIXES`)
  - `BLACKLIST_RE` for filtering abstract/data-object terms
- EU institution vocabulary used by `PhraseMatcher` (`EU_INSTITUTIONS`).
- GLiNER config (`GLINER_LABELS`, defaults).
- Default models (`DEFAULT_MODEL`, `DEFAULT_SPACY_MODEL`, …).
- Prompt templates for:
  - actor extraction (`ACTOR_EXTRACTION_PROMPT`)
  - hierarchy extraction (`HIERARCHY_EXTRACTION_PROMPT`)
  - hybrid validation prompt (`PRE_EXTRACTED_ACTORS_IDENTIFIED`)

---

### `src/pipeline/organization/models.py`
Typed data structures used across the pipeline.

- `OrganizationalEntity`: represents a UNIT or ROLE and its parents.
- `Subject`: represents a person placeholder bound to a unit+role (CPEE requires this section).

---

### `src/pipeline/organization/utils.py`
Shared helpers for prompt construction, text normalization, and XML generation.

Includes:
- Prompt builders:
  - `_build_actor_extraction_prompt(text)`
  - `_build_hierarchy_extraction_prompt(actor_lines, text)`
  - `_build_pre_extracted_actors_hierarchy_prompt(candidate_lines, text)`
- Text helpers:
  - `normalize_name()` (strips determiners like “the”, normalizes whitespace)
  - `is_actor()` (applies blacklist)
  - `infer_type()` (suffix heuristic)
- XML serialization:
  - `_generate_organigram_xml(entities, subjects, output_path)`
  - `create_xml(root, output_path)`

---

## Output format (`organigram.xml`)

The generated file is a CPEE-style organization model with:

- `<units>`: UNIT entities + their `<parent>` relationships
- `<roles>`: ROLE entities + their `<parent>` relationships
- `<subjects>`: dummy subjects linked via `<relation unit="..." role="..." />`

This output can be used as the organizational model for process execution / BPMN/CPEE tooling and as structured input for later extraction steps.
