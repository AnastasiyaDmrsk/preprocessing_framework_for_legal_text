# Master Thesis: Automated Extraction of Organizational Information and Process Descriptions from Regulatory Documents

## Approach

### Preprocessing

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

The evaluation dataset is located in `eval/dataset`.

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