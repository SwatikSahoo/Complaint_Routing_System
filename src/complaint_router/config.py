from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
COMPLAINTS_CSV = DATA_DIR / 'complaints_dataset.csv'
OFFICERS_CSV = DATA_DIR / 'officers_dataset.csv'
HISTORICAL_CSV = DATA_DIR / 'historical_complaints_dataset.csv'
VECTORIZER_PATH = MODELS_DIR / 'vectorizer.joblib'
PRIORITY_MODEL_PATH = MODELS_DIR / 'priority_model.joblib'
ETA_MODEL_PATH = MODELS_DIR / 'eta_model.joblib'
ROUTER_MODEL_PATH = MODELS_DIR / 'router_model.joblib'
SIMILARITY_INDEX_PATH = MODELS_DIR / 'similarity_index.joblib'
METADATA_PATH = MODELS_DIR / 'metadata.joblib'
EVAL_PATH = MODELS_DIR / 'evaluation.json'
