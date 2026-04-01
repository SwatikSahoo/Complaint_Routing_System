from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

from src.complaint_router.config import (
    COMPLAINTS_CSV,
    OFFICERS_CSV,
    HISTORICAL_CSV,
    MODELS_DIR,
    METADATA_PATH,
    PRIORITY_MODEL_PATH,
    ETA_MODEL_PATH,
    ROUTER_MODEL_PATH,
    SIMILARITY_INDEX_PATH,
    VECTORIZER_PATH,
    EVAL_PATH,
)
from src.complaint_router.data import load_datasets
from src.complaint_router.features import build_training_text, fit_vectorizer, transform_text, build_numeric_features, combine_features
from src.complaint_router.modeling import (
    train_priority_classifier,
    train_eta_regressor,
    train_router_model,
    build_similarity_index,
    evaluate_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train complaint routing system.')
    parser.add_argument('--complaints', type=Path, default=COMPLAINTS_CSV)
    parser.add_argument('--officers', type=Path, default=OFFICERS_CSV)
    parser.add_argument('--historical', type=Path, default=HISTORICAL_CSV)
    parser.add_argument('--models-dir', type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args.complaints, args.officers, args.historical)
    complaints = datasets.complaints.copy()

    train_df, test_df = train_test_split(
        complaints,
        test_size=0.2,
        random_state=42,
        stratify=complaints['priority_label'],
    )

    train_text = build_training_text(train_df)
    test_text = build_training_text(test_df)
    hist_text = build_training_text(
        datasets.historical.rename(columns={'historical_complaint_id': 'complaint_id'}).assign(input_type='text', attachments_count=0, citizen_sentiment_score=0)
    )

    vectorizer = fit_vectorizer(train_text)
    X_train_text = transform_text(vectorizer, train_text)
    X_test_text = transform_text(vectorizer, test_text)
    X_hist_text = transform_text(vectorizer, hist_text)

    X_train_num = build_numeric_features(train_df)
    X_test_num = build_numeric_features(test_df)
    X_train = combine_features(X_train_text, X_train_num)
    X_test = combine_features(X_test_text, X_test_num)

    priority_model = train_priority_classifier(X_train, train_df['priority_label'])
    eta_model = train_eta_regressor(X_train, train_df['eta_days_label'])
    router_model = train_router_model(X_train, train_df['assigned_officer_id'])
    similarity_index = build_similarity_index(X_hist_text, n_neighbors=5)

    evaluation = evaluate_models(
        priority_model=priority_model,
        eta_model=eta_model,
        router_model=router_model,
        X_test=X_test,
        X_test_dense=X_test.toarray(),
        y_priority_test=test_df['priority_label'],
        y_eta_test=test_df['eta_days_label'],
        y_router_test=test_df['assigned_officer_id'],
    )

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(priority_model, PRIORITY_MODEL_PATH)
    joblib.dump(eta_model, ETA_MODEL_PATH)
    joblib.dump(router_model, ROUTER_MODEL_PATH)
    joblib.dump(similarity_index, SIMILARITY_INDEX_PATH)
    joblib.dump(
        {
            'officers': datasets.officers,
            'historical': datasets.historical,
            'evaluation': evaluation,
        },
        METADATA_PATH,
    )
    EVAL_PATH.write_text(json.dumps(evaluation, indent=2))

    print('Training completed successfully.')
    print(json.dumps(evaluation, indent=2))


if __name__ == '__main__':
    main()
