# Process Description Generation (Role→Task Mapping → Textual Process)

This module turns **(1) preprocessed regulatory text** plus **(2) a role–task mapping (XML)** into a **structured natural-language process description** that mirrors a BPMN collaboration description.

It is designed as a lightweight pipeline:

1. Parse and sort tasks from the role–task mapping XML.
2. Serialize tasks into a compact prompt-friendly representation.
3. Ask an LLM to generate a linear, actor-structured process narrative.
4. (Optional but recommended) ask a validator LLM to enforce completeness, ordering, and modality.
5. Save the final description to disk (default: `process_description.txt`).

---

## Output

- A **plain-text** process description saved as `process_description.txt` in the job output directory.
- The description:
  - declares actors (pools/lanes conceptually),
  - describes a start trigger,
  - provides a per-actor linear narrative of activities,
  - includes gateways (XOR/AND) using explicit phrasing,
  - preserves deontic modality (`shall`, `may`, etc.),
  - includes conditions and exceptions.

---

## Inputs
1) `preprocessed_text: str`
A cleaned / preprocessed version of the regulatory text.
   - This module treats the text as the authoritative source for **activity ordering**, gateway cues, and flow.
   - The role–task mapping is treated as the authoritative source for **who does what**.
2) `role_task_mapping: Union[str, Path]`
An XML document describing tasks, performers (actors), modality, conditions, exceptions, and source references.

---

## Task extraction using NLP

The task extraction is implemented in `src/pipeline/process_description/nlp_extractor.py`.

Extraction behavior:
- `task_id`: from `<task id="…">` (fallback `"unknown"`)
- `label`: from `<label>` text
- `deontic_modality`: from `<deontic modality="…">` (fallback `"shall"`)
- `actors`: unique list of `<performer><role>…</role></performer>`
- `conditions`: all `<condition>` texts
- `exceptions`: all `<exception description="…">` strings
- `source_article` / `source_paragraph`: from `<source-ref article="…" paragraph="…"/>` (fallback `"UNKNOWN"`)

If XML parsing fails, a warning is emitted and an empty list is returned.

#### `ProcessDescriptionNLPExtractor.extract_tasks(xml_content: str) -> List[ExtractedTask]`
Returns tasks sorted by:

1. article number (numeric prefix; unknown articles sort last)
2. paragraph (supports formats like `"1"` or `"1(a)"`)
3. numeric part of `task_id` (e.g., `t12`)

Important: This ordering is only a *stable baseline*. The prompt instructs the LLM to follow the **regulatory text order** as the primary order signal.

---

## Prompting and LLM interaction

The LLM generation is implemented in `src/pipeline/process_description/llm_generator.py`.
It uses `google.genai` to call a configured model.

Key methods:
- `generate_and_save(preprocessed_text, tasks, output_path) -> str`
  - serializes tasks,
  - generates a drafted process description,
  - optionally validates it,
  - saves output to `output_path`.

- `_generate(...) -> str`
  - fills `PROCESS_DESCRIPTION_PROMPT` with:
    - `[ROLE TASK MAPPING]` (serialized tasks)
    - `[REGULATORY TEXT]` (preprocessed text)

- `_validate(...) -> str`
  - fills `PROCESS_DESCRIPTION_VALIDATION_PROMPT` with:
    - the same inputs plus `[PROPOSED_DESCRIPTION]`
  - expects the validator to return the final text wrapped in:
    - `<FINAL_DESCRIPTION> ... </FINAL_DESCRIPTION>`
  - if tags are missing, it falls back to the generator output and emits a warning.

---

## Error handling and known gotchas

- XML parsing errors are handled with warnings and return an empty task list.
- Validator responses must include `<FINAL_DESCRIPTION>` tags; otherwise the generator output is used.
