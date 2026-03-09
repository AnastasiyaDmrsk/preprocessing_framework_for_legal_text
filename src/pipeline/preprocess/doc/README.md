# Preprocessing

This folder documents the preprocessing pipeline in `src/pipeline/preprocess/`. The goal is to transform **EU regulatory text into a structured, cleaned representation** that is particularly suitable for **downstream semantic extraction** (e.g., BPMN / process extraction).

The pipeline focuses on:

- **Structure detection** (articles/sections) and segmentation
- **Sentence/paragraph parsing** including enumerations (list items) and gateways
- **Noise / filler removal** (static patterns and syntactically detected filler phrases)
- **Extraction of (internal/external) references** into a separate list/CSV
- **Subordinate clause reduction** (optional via Benepar, regex fallback)
- **Heuristic actor / pronoun / passive normalization**
- **Filtering for “obligation” sentences** based on deontic modals (`shall`, `must`, ...)

> Note: This preprocessing implementation is **not** a generic tokenize/lemmatize/stopword pipeline. It is a **regulation-specific** preprocessor designed to isolate “obligation clauses” as cleanly as possible.

---

## Quick usage

### Programmatic API

The easiest way to use it is via `src/pipeline/preprocess/api.py`:

- `preprocess_legal_text(reg_text: str, path: Path) -> (preprocessed_text, references)`
- writes outputs into the target folder `path`

Outputs:

- `preprocess.txt` – **one obligation clause per line**, with gateway markers
- `references.csv` – CSV with `Location,Reference` for detected cross-references

### Direct call

`RegulatoryTextPreprocessor` can also be used directly:

- `preprocessed_text, references = RegulatoryTextPreprocessor().preprocess(text)`

---

## Output format (`preprocess.txt`)

The pipeline produces a line-based text that is suitable as input for downstream extractors.

Typical lines:

- `Article 23: Reporting obligations` (heading)
- `1 Each Member State shall ensure ...` (paragraph + obligation sentence)
- `---` (separator between articles)
- `--PARALLEL GATEWAY--` / `--END OF PARALLEL GATEWAY--` (markers for parallel enumerations)

List items are often emitted as separate lines and (when detected) prefixed with their list key (e.g., `a`, `1(a)` etc.).

---

## Traceability / references

In this implementation, “traceability” is primarily represented as follows:

1. **Location keys** are carried through into the output (e.g., `Article 23.1`, `Article 23.4 (1(a))`).
2. **References** (EU acts and internal references like “Article 12”) are written as a **separate table** in `references.csv`.

Important: references are extracted **before** removing the corresponding spans. The citation can then be removed from the sentence without losing the information.

---

## Pipeline overview

Implemented in `src/pipeline/preprocess/preprocess.py`.

### 1) spaCy setup + optional Benepar integration

On initialization it:

- loads a spaCy model (`en_core_web_md` preferred, fallback to `en_core_web_sm`)
- optionally tries to enable `benepar` (constituency parser)
- initializes matchers for references

Relevant: `__init__()`, `_parse_plain()`, `_init_matchers()`.

**Benepar is optional**:

- if available: subordinate clause removal via constituency trees
- if not: regex fallback

### 2) Detect document structure

`_detect_document_structure()` checks via regex:

- standard article structure: `Article <n>` (`ARTICLE_STANDARD_RE`)
- alternative article structure: `Art. <n>` (`ARTICLE_ALT_RE`)
- numbered sections: `1.2.3 ...` (`SECTION_NUMBERING_RE`)

Depending on the structure, it dispatches to a handler:

- `_preprocess_standard_articles()`
- `_preprocess_alt_articles()`
- `_preprocess_section_numbering()`
- `_preprocess_raw_text()`

### 3) Article / paragraph / list splitting

For articles, text spans are extracted and then split into paragraphs and (if present) list items.

- `_split_*articles()` – splitting by article headers
- `_parse_paragraphs_with_lists()` – detects e.g. `(a) ... (b) ...` or `a) ...` and emits:
  - **gateway markers** for parallel paths
  - one “pseudo-sentence” per list item (including the “leading” context)

### 4) Sentence processing (“clause” cleaning)

Every sentence/paragraph is processed by `_process_sentence()`.

Flow (simplified):

1. **Remove static fillers** (`apply_static_fillers`) and normalize whitespace
2. **Remove subordinate clauses**
   - if `benepar` is available: via `_plan_subordinate_removal()` / `_collect_subordinate_spans()`
   - else: `_remove_subordinate_clauses_regex()`
3. **Extract references** (`_extract_references_from_doc`) and remove reference spans
4. Remove additional dynamic fillers (`plan_filler_removal`, e.g. “where ...,” / “including ...”)
5. **Heuristically normalize actor / pronouns / passive voice**
   - `extract_explicit_actor()`
   - `plan_pronoun_resolution()` (“it/they” -> actor)
   - `plan_passive_resolution()` (agentless passive -> “by <actor>”)
6. Apply token plan (`TokenTransformPlan.apply()`)
7. Clean external reference phrases (`remove_external_reference_phrases`)
8. **Obligation filter**: if no modal is present (`OBLIGATION_RE`), the sentence is dropped (`None`)
9. Postprocessing:
   - gateway highlights: `highlight_or_between_verbs()` / `highlight_and_between_modal_verbs()`
   - IF normalization (`normalize_if`) and `in accordance with` -> `IAW` (`apply_iaw`)

Return value: `(processed_sentence, explicit_actor)` or `None` (if not relevant).

---

## Key heuristics / guards

### “Subordinate clause removal” guards

When removing SBAR/WH* subordinate spans, guards are applied (see the docstring in `_collect_subordinate_spans()`):

- **G1**: deontic modal present in the clause -> keep
- **G2**: clause starts with a conditional subordinator (“if”, “unless”, ...) -> keep
- **G3**: internal reference present in the clause -> keep
- **G5**: clause contains actor + activity -> keep

External references (EU act citations) are *not* a guard because they are extracted beforehand.

---

## Configuration / customization

The central constants live in `src/pipeline/preprocess/const.py`:

- structure regexes (`ARTICLE_STANDARD_RE`, ...)
- deontic modals (`DEONTIC_MODALS`, `OBLIGATION_RE`)
- filler patterns (`STATIC_FILLER_PATTERNS`)
- gateway markers (`PARALLEL_GATEWAY_START/END`)
- actor heuristics (`LEGAL_ENTITY_KEYWORDS`, `ACTOR_IGNORE`, `ACTOR_NER_LABELS`)
- Benepar token limit (`_BENEPAR_MAX_TOKENS`)

---

## File overview

- `preprocess.py`
  - `RegulatoryTextPreprocessor`: end-to-end pipeline
  - structure detection, article/paragraph splitting, sentence processing, reference extraction

- `api.py`
  - `preprocess_legal_text(...)`: convenience API, writes `preprocess.txt` and `references.csv`

- `nlp_utils.py`
  - syntactic heuristics: filler removal, actor extraction, pronoun/passive resolution
  - gateway highlighting
  - removal of external reference phrases (depending on which articles exist in the document)

- `text_utils.py`
  - text normalization (whitespace, “IF”, “IAW”)
  - `TokenTransformPlan`: collect token edits and reconstruct the sentence in one pass
  - `build_eu_ref_matcher`: rule-based detection of EU act citations (spaCy matcher)

- `const.py`
  - regexes, vocabularies, and markers (configuration)

---

## Limits / known trade-offs

- Sentence segmentation uses spaCy `Doc.sents`. For highly “legalese” enumerations, segmentation can vary depending on the input.
- The pipeline focuses on **obligation clauses**: sentences without a deontic modal are dropped.
- Token-based edits are intentionally heuristic and may oversimplify in edge cases.

---

## Useful next extensions (optional)

- “True” traceability: map output lines back to input offsets (character spans), not only location keys.
- Persisted configuration (e.g., YAML) for modals/fillers/keywords.
- Unit tests for structure splitting, reference extraction, and guard logic.
