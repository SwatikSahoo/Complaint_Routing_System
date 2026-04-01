# Complaint Auto-Routing System (CLI)

Offline, end-to-end Python project for the **AI/ML Dev - Assignment: Complaint Auto-Routing System**. It uses only local Python libraries and the provided synthetic datasets.

## What this project does
- Officer routing: recommends the best officer(s)
- Priority prediction: predicts `high`, `medium`, or `low`
- ETA estimation: predicts resolution time in days
- Similarity search: retrieves related historical complaints
- Multilingual handling: trains on `normalized_text_en` while preserving original multilingual inputs in the data
- Audio/video handling: CLI accepts audio/video as **transcribed text**, which is a practical offline design for the assignment

## Project structure

```text
complaint_router_cli/
├── cli.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
│   ├── complaints_dataset.csv
│   ├── officers_dataset.csv
│   └── historical_complaints_dataset.csv
├── models/
└── src/
    └── complaint_router/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── features.py
        ├── modeling.py
        └── pipeline.py
```

## Modeling approach

### 1) Text representation
The training pipeline builds a combined text field from:
- normalized complaint text
- city
- ward
- category
- subcategory
- language
- input type

It then creates TF-IDF vector features with uni-grams and bi-grams. Numeric features like sentiment score and attachment count are concatenated to the sparse feature matrix.

### 2) Priority model
- Model: `LogisticRegression`
- Target: `priority_label`
- Metrics: accuracy and weighted F1

### 3) ETA model
- Model: `RandomForestRegressor`
- Target: `eta_days_label`
- Metric: MAE

### 4) Officer routing
- Model: `LogisticRegression`
- Target: `assigned_officer_id`
- Final officer ranking = router probability + business rules

Business rules used for re-ranking:
- category specialization match
- city match
- language support match
- ward match
- officer performance score
- average resolution speed
- active case capacity

### 5) Similarity search
- Features: historical complaint TF-IDF vectors
- Index: `NearestNeighbors(metric="cosine")`
- Output: top-k similar historical complaints with similarity score

## Why audio/video are handled as transcripts
The assignment asks for multilingual text plus audio/video, but the downstream ML tasks are routing, priority, ETA, and similarity. In a production-safe offline baseline, audio/video are converted to text first using local speech-to-text. This repo keeps the interface ready for that by accepting:
- `--input-type audio` or `--input-type video`
- `--text` or `--transcript-file`

That keeps the ML system unified and avoids maintaining separate models per modality.

## Setup

### 1) Create a virtual environment
```bash
python -m venv .venv
```

Activate it.

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Train models
```bash
python train.py
```

This writes all trained artifacts into `models/`.

### 4) Run evaluation summary
```bash
python evaluate.py
```

### 5) Run inference from CLI

**Windows**
```bash
python cli.py --text "Water pipeline leakage near market road for two days" --category water --subcategory "pipeline leakage" --city Bhubaneswar --ward Ward-08 --language en --attachments-count 2 --citizen-sentiment-score 0.35
```

```bash
python cli.py \
  --text "Water pipeline leakage near market road for two days" \
  --category water \
  --subcategory "pipeline leakage" \
  --city Bhubaneswar \
  --ward Ward-08 \
  --language en \
  --attachments-count 2 \
  --citizen-sentiment-score 0.35
```

### Example for audio complaint using transcript text
```bash
python cli.py \
  --input-type audio \
  --text "There is a dangerous open manhole near Ward-08 after rain" \
  --category drainage \
  --subcategory "manhole cover broken" \
  --city Bhubaneswar \
  --ward Ward-08 \
  --language en
```

### JSON output mode
```bash
python cli.py \
  --text "Street light not working for three nights" \
  --category electricity \
  --subcategory "street light outage" \
  --city Cuttack \
  --ward Ward-04 \
  --language en \
  --json
```

## Suggested interview explanation
“I built a unified offline ML pipeline where multilingual text, and audio/video transcripts, are normalized into a common text representation. I trained a multi-task system for priority classification, ETA regression, officer routing, and historical similarity search, then combined learned routing probabilities with business constraints like specialization, city, ward, and language support to produce practical officer recommendations.”

## Future improvements
- Replace TF-IDF with a fully local sentence embedding model when model files are available offline
- Add Vosk or Whisper local STT integration for direct audio/video processing
- Add confidence calibration and feedback loop re-training
- Add FastAPI or Streamlit frontend
