# Overview

There are two working pipelines for Task 2.

The goal is to provide a clear and reusable pipeline for experimentation, rather than a final model.

---

## 1. What is implemented

### 1.1 Token-level extraction (baseline)

A standard sequence tagging pipeline:

- Input tokens → embedding (random or GloVe)
- BiLSTM encoder
- Token-level classification (BIO labels)

This is a simple end-to-end extraction model.

---

### 1.2 Decomposed pipeline (sentence filtering + extraction)

A two-step pipeline:

Step 1: Sentence filtering
- TF-IDF + Logistic Regression
- Predicts whether a sentence contains relevant information

Step 2: Token extraction
- Apply BiLSTM tagger only to selected sentences
- Reconstruct documents from filtered sentences

This allows comparison between:
- End-to-end vs decomposed pipelines

---

## 2. How to run

Main experiments are in main.py:

The code should run in **text_analytics** environment

---

## 3. Code structure (quick overview)

Main entry:
- main.py → runs experiments

Pipelines:
- separate_pipeline.py → token-level pipeline
- sentence_filtering.py → sentence filtering
- joint_pipeline.py → old joint version (not used)

Model:
- lstm_model.py → BiLSTM model + training + prediction

Data:
- data_utils.py → data loading + BIO conversion
- dataloader_utils.py → encoding + padding + dataloaders

Evaluation:
- evaluate.py → token-level classification_report

---

## 4. Important parameters / settings

### Target field

Default:
label_type = 'participants'

Can be changed to:
participants / interventions / outcomes

---

### Pretrained embeddings 

If `use_pre_trained = 1`, you need to download the GloVe embeddings manually.

Download from:
https://nlp.stanford.edu/projects/glove/

Use:
glove.6B.50d.txt

Place it in:
./glove.6B/

So the final path should be:
./glove.6B/glove.6B.50d.txt

If not available, set:
use_pre_trained = 0

---

### Sequence length / runtime

max_len = 400

- used for fast testing

If set to None:
- uses full sequence length
- runtime ≈ 20–30 minutes

---

### Model hyperparameters

- embedding dim: 50
- hidden dim: 64
- labels: 3 (BIO)

---

### Training settings

- epochs: 3
- learning rate: 1e-3
- batch size: 32
- optimizer: Adam

---

### Loss function

Defined in lstm_model.py.

Options:
- CrossEntropyLoss
- Weighted CrossEntropyLoss
- Focal Loss

Only one should be used at a time.

---

### Sentence filtering

- TF-IDF (1–2 grams)
- Logistic Regression (balanced)

Settings:
use_gold_sent = False  
fallback_top1 = True

---

### Data path (important)

DATA_DIR = "../ebm_nlp_2_00"
change it to your own data path
---

### Evaluation

- token-level
- classification_report

---

## 5. Joint pipeline (not used)

A joint pipeline (multi-field tagging) was implemented but is not used because:

- does not match original data structure
- tokens may belong to multiple fields
- requires artificial priority rules
- reduces interpretability

So we focus on separate extraction per field.

---

## 6. Possible extensions

- add CRF
- span-level evaluation
- better sentence filtering

---
