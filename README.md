# Master Thesis: Automated Extraction of Organizational Information and Process Descriptions from Regulatory Documents

## Approach

### Preprocessing

### Organizational Information Extraction and Organigram Generation

### Role-Task Mapping

### Process Description Generation

## Evaluation

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