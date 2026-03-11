# Master Thesis: Automated Extraction of Organizational Information and Process Descriptions from Regulatory Documents

## Approach

### Preprocessing

This step prepares regulatory/legal text for downstream extraction. The current implementation is tailored to EU-style regulatory documents and focuses on producing a clean, structure-aware representation of *obligation clauses*.

In short, the pipeline:
- detects document structure (e.g., `Article` / `Art.` / numbered sections) and splits text into article/paragraph units,
- handles common legal enumerations (e.g., `(a) ... (b) ...`) and emits gateway markers for parallel list items,
- removes boilerplate/filler phrases and optionally reduces removable subordinate clauses (Benepar if available; regex fallback),
- extracts internal/external references into a separate `references.csv` for traceability,
- applies lightweight linguistic normalization (actor/pronoun/passive heuristics) and filters to sentences containing deontic modals (e.g., *shall*, *must*).

For implementation details and outputs, see the preprocessing documentation: [Preprocessing](src/pipeline/preprocess/doc/README.md).

---

### Organizational Information Extraction and Organigram Generation

This step implements an **organizational actor extraction** task that turns **preprocessed legal text** into a **CPEE-compatible `organigram.xml`**. The design supports two modes:

- **Pure LLM extraction**: the LLM extracts actors and their hierarchies directly.
- **Hybrid extraction (default & recommended)**: NLP heuristics extract candidate actors first, then the LLM validates/corrects them and infers hierarchies.

The goal is to extract:

- **Actors** as either **UNIT** (institutions/bodies) and/or **ROLE** (functions/positions)
- **Hierarchies**: UNIT-UNIT, ROLE-ROLE, and ROLE-UNIT relations

**Input**: `preprocessed_text` created in the previous step.

**Output**: a structured XML output (`organigram.xml`) suitable for downstream BPMN/CPEE tooling.

For more details on the design and implementation, see the [Organizational Information Extraction & Organigram Generation documentation](src/pipeline/organization/doc/organizational_information.md).

---


### Role-Task Mapping

---

### Process Description Generation

## Evaluation

The evaluation dataset is located in `eval/dataset`. The dataset includes following regulatory documents:
1. [AI Act Regulation](http://data.europa.eu/eli/reg/2024/1689/oj)
2. [GDPR Regulation](http://data.europa.eu/eli/reg/2016/679/oj)
3. [Driving Licences Directive](http://data.europa.eu/eli/dir/2025/2205/oj)
4. [Health Data Regulation](http://data.europa.eu/eli/reg/2013/604/oj)
5. [CDD Regulation](http://data.europa.eu/eli/dir/2015/849/oj)
6. [eIDAS Regulation](http://data.europa.eu/eli/reg/2014/910/oj)
7. [NIS2 Directive](http://data.europa.eu/eli/dir/2022/2555/oj)
8. [Medical Device Regulation](http://data.europa.eu/eli/reg/2017/745/oj)
9. [Digital Services Act (DSA) Regulation](http://data.europa.eu/eli/reg/2022/2065/oj)
10. [Cybersecurity Act Regulation](http://data.europa.eu/eli/reg/2019/881/oj)

All gold standards can be found as follows:
- **BPMN gold standards**: `eval/1_preprocessing_eval/gold_standard`
- **Organigram gold standards**: `eval/2_3_organigram_eval/gold_standard`
- **Role-task mapping gold standards**: `eval/2_3_organigram_eval/gold_standard_mapping`

### Preprocessed Text Evaluation



### Organigram Evaluation
Make sure to install: 
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```
Single file vs single file evaluation:
```bash
python evaluate_organigrams.py \
    --gold gold_standard/5_CDD.xml \
    --pred results/5_CDD.xml \
    --out evaluation_results_5_CDD.csv \
    --threshold 0.82
```
Folder vs folder evaluation:
```bash
python evaluate_organigrams.py \
    --gold gold_standard \
    --pred results \
    --out evaluation_results.csv \
    --threshold 0.82
```

### Role-Task Mapping Evaluation

### Process Description Evaluation