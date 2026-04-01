from __future__ import annotations

from typing import Iterable
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

WORD_RE = re.compile(r'[^a-z0-9\s-]+', flags=re.IGNORECASE)

def clean_text(text: str) -> str:
    text = (text or '').strip().lower()
    text = WORD_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_training_text(df: pd.DataFrame) -> pd.Series:
    parts = (
        df['normalized_text_en'].fillna('')
        + ' city:' + df['city'].fillna('')
        + ' ward:' + df['ward'].fillna('')
        + ' category:' + df['category'].fillna('')
        + ' subcategory:' + df['subcategory'].fillna('')
        + ' language:' + df['language'].fillna('')
        + ' input_type:' + df['input_type'].fillna('text')
    )
    return parts.map(clean_text)

def fit_vectorizer(texts: Iterable[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
        strip_accents='unicode',
    )
    vectorizer.fit(texts)
    return vectorizer

def transform_text(vectorizer: TfidfVectorizer, texts: Iterable[str]):
    cleaned = [clean_text(t) for t in texts]
    return vectorizer.transform(cleaned)

def build_numeric_features(df: pd.DataFrame) -> csr_matrix:
    cols = []
    for col in ['citizen_sentiment_score', 'attachments_count']:
        if col in df.columns:
            cols.append(pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy().reshape(-1, 1))
    if not cols:
        return csr_matrix((len(df), 0))
    arr = np.hstack(cols)
    return csr_matrix(arr)

def combine_features(text_matrix, numeric_matrix):
    if numeric_matrix.shape[1] == 0:
        return text_matrix
    return hstack([text_matrix, numeric_matrix]).tocsr()
