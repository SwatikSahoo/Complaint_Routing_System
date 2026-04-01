from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors



def train_priority_classifier(X_train: csr_matrix, y_train: pd.Series) -> SGDClassifier:
    model = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=2000, tol=1e-3)
    model.fit(X_train, y_train)
    return model



def train_eta_regressor(X_train: csr_matrix, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=18,
        random_state=42,
        min_samples_leaf=1,
        n_jobs=-1,
    )
    model.fit(X_train.toarray(), y_train)
    return model



def train_router_model(X_train: csr_matrix, y_train: pd.Series) -> Dict[str, object]:
    officer_ids: List[str] = []
    rows = []
    y_train = y_train.reset_index(drop=True)
    for officer_id in sorted(y_train.unique()):
        idx = np.where(y_train.to_numpy() == officer_id)[0]
        proto = X_train[idx].mean(axis=0)
        rows.append(csr_matrix(proto))
        officer_ids.append(officer_id)
    prototype_matrix = vstack(rows)
    return {'officer_ids': officer_ids, 'prototype_matrix': prototype_matrix}



def router_predict(router_model: Dict[str, object], X: csr_matrix) -> np.ndarray:
    sims = cosine_similarity(X, router_model['prototype_matrix'])
    best_idx = sims.argmax(axis=1)
    officer_ids = np.array(router_model['officer_ids'])
    return officer_ids[best_idx]



def router_predict_proba(router_model: Dict[str, object], X: csr_matrix) -> np.ndarray:
    sims = cosine_similarity(X, router_model['prototype_matrix'])
    sims = np.maximum(sims, 0)
    row_sums = sims.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return sims / row_sums



def build_similarity_index(X_hist: csr_matrix, n_neighbors: int = 5) -> NearestNeighbors:
    index = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    index.fit(X_hist)
    return index



def evaluate_models(
    priority_model,
    eta_model,
    router_model,
    X_test,
    X_test_dense,
    y_priority_test,
    y_eta_test,
    y_router_test,
) -> Dict[str, float]:
    priority_pred = priority_model.predict(X_test)
    eta_pred = eta_model.predict(X_test_dense)
    router_pred = router_predict(router_model, X_test)

    return {
        'priority_accuracy': float(accuracy_score(y_priority_test, priority_pred)),
        'priority_f1_weighted': float(f1_score(y_priority_test, priority_pred, average='weighted')),
        'eta_mae_days': float(mean_absolute_error(y_eta_test, eta_pred)),
        'routing_accuracy': float(accuracy_score(y_router_test, router_pred)),
        'routing_f1_weighted': float(f1_score(y_router_test, router_pred, average='weighted')),
    }
