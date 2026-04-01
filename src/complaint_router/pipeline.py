from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .features import clean_text, combine_features
from .modeling import router_predict_proba


@dataclass
class ComplaintInput:
    text: str
    category: str
    subcategory: str
    city: str
    ward: str
    language: str = 'en'
    input_type: str = 'text'
    attachments_count: int = 0
    citizen_sentiment_score: float = 0.0


class ComplaintRoutingPipeline:
    def __init__(self, artifacts_dir):
        self.vectorizer = joblib.load(artifacts_dir / 'vectorizer.joblib')
        self.priority_model = joblib.load(artifacts_dir / 'priority_model.joblib')
        self.eta_model = joblib.load(artifacts_dir / 'eta_model.joblib')
        self.router_model = joblib.load(artifacts_dir / 'router_model.joblib')
        self.similarity_index = joblib.load(artifacts_dir / 'similarity_index.joblib')
        self.metadata = joblib.load(artifacts_dir / 'metadata.joblib')
        self.officers = self.metadata['officers'].copy()
        self.historical = self.metadata['historical'].copy()

    def _to_frame(self, complaint: ComplaintInput) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'normalized_text_en': complaint.text,
                'category': complaint.category,
                'subcategory': complaint.subcategory,
                'city': complaint.city,
                'ward': complaint.ward,
                'language': complaint.language,
                'input_type': complaint.input_type,
                'citizen_sentiment_score': complaint.citizen_sentiment_score,
                'attachments_count': complaint.attachments_count,
            }
        ])

    def _build_feature_row(self, frame: pd.DataFrame):
        text = (
            frame['normalized_text_en'].fillna('')
            + ' city:' + frame['city'].fillna('')
            + ' ward:' + frame['ward'].fillna('')
            + ' category:' + frame['category'].fillna('')
            + ' subcategory:' + frame['subcategory'].fillna('')
            + ' language:' + frame['language'].fillna('')
            + ' input_type:' + frame['input_type'].fillna('text')
        ).map(clean_text)
        text_matrix = self.vectorizer.transform(text)
        numeric = csr_matrix(
            np.array([
                [
                    float(frame.iloc[0].get('citizen_sentiment_score', 0.0) or 0.0),
                    float(frame.iloc[0].get('attachments_count', 0) or 0),
                ]
            ])
        )
        return combine_features(text_matrix, numeric), text_matrix

    def _rank_officers(self, complaint: ComplaintInput, routing_proba: np.ndarray) -> List[Dict]:
        classes = list(self.router_model['officer_ids'])
        class_to_prob = {cls: float(prob) for cls, prob in zip(classes, routing_proba)}

        candidates = self.officers.copy()
        candidates = candidates[candidates['specialization_category'].str.lower() == complaint.category.lower()]
        if candidates.empty:
            candidates = self.officers.copy()

        same_city = candidates[candidates['city'].str.lower() == complaint.city.lower()]
        if not same_city.empty:
            candidates = same_city

        def language_match(langs: str) -> float:
            supported = {x.strip().lower() for x in str(langs).split(',') if x.strip()}
            return 1.0 if complaint.language.lower() in supported else 0.0

        def ward_match(primary_ward: str) -> float:
            return 1.0 if str(primary_ward).strip().lower() == complaint.ward.lower() else 0.0

        candidates['router_probability'] = candidates['officer_id'].map(lambda oid: class_to_prob.get(oid, 0.0))
        candidates['language_match'] = candidates['languages_supported'].map(language_match)
        candidates['ward_match'] = candidates['primary_ward'].map(ward_match)
        candidates['performance_norm'] = candidates['performance_score'] / max(candidates['performance_score'].max(), 1.0)
        candidates['capacity_norm'] = 1.0 - (candidates['active_cases_capacity'] / max(candidates['active_cases_capacity'].max(), 1.0))
        candidates['eta_norm'] = 1.0 - (candidates['avg_resolution_days'] / max(candidates['avg_resolution_days'].max(), 1.0))

        candidates['final_score'] = (
            0.45 * candidates['router_probability']
            + 0.20 * candidates['performance_norm']
            + 0.10 * candidates['language_match']
            + 0.10 * candidates['ward_match']
            + 0.10 * candidates['eta_norm']
            + 0.05 * candidates['capacity_norm']
        )

        cols = [
            'officer_id', 'officer_name', 'department', 'specialization_category',
            'city', 'primary_ward', 'languages_supported', 'avg_resolution_days',
            'performance_score', 'final_score'
        ]
        return candidates.sort_values('final_score', ascending=False)[cols].head(3).to_dict(orient='records')

    def _similar_complaints(self, text_matrix, top_k: int = 3) -> List[Dict]:
        distances, indices = self.similarity_index.kneighbors(text_matrix, n_neighbors=top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.historical.iloc[int(idx)]
            results.append({
                'historical_complaint_id': row['historical_complaint_id'],
                'text': row['original_text'],
                'normalized_text_en': row['normalized_text_en'],
                'category': row['category'],
                'priority_label': row['priority_label'],
                'actual_resolution_days': int(row['actual_resolution_days']),
                'resolution_summary': row['resolution_summary'],
                'similarity_score': float(1.0 - dist),
            })
        return results

    def predict(self, complaint: ComplaintInput) -> Dict:
        frame = self._to_frame(complaint)
        X_row, text_matrix = self._build_feature_row(frame)

        priority = self.priority_model.predict(X_row)[0]
        priority_proba = self.priority_model.predict_proba(X_row)[0]
        eta_days = float(self.eta_model.predict(X_row.toarray())[0])
        routing_proba = router_predict_proba(self.router_model, X_row)[0]
        officers = self._rank_officers(complaint, routing_proba)
        similar = self._similar_complaints(text_matrix, top_k=3)

        return {
            'input': complaint.__dict__,
            'predictions': {
                'priority': str(priority),
                'priority_confidence': float(np.max(priority_proba)),
                'eta_days': max(1, int(round(eta_days))),
                'recommended_officers': officers,
                'similar_complaints': similar,
            }
        }

    @staticmethod
    def pretty_print(result: Dict) -> str:
        lines = []
        lines.append('=== Complaint Auto-Routing Result ===')
        lines.append(f"Priority: {result['predictions']['priority']} (confidence={result['predictions']['priority_confidence']:.2f})")
        lines.append(f"Estimated Resolution Time: {result['predictions']['eta_days']} day(s)")
        lines.append('')
        lines.append('Top Officer Recommendations:')
        for i, officer in enumerate(result['predictions']['recommended_officers'], start=1):
            lines.append(
                f"  {i}. {officer['officer_name']} [{officer['officer_id']}] | {officer['department']} | "
                f"Ward: {officer['primary_ward']} | Score: {officer['final_score']:.3f}"
            )
        lines.append('')
        lines.append('Most Similar Historical Complaints:')
        for item in result['predictions']['similar_complaints']:
            lines.append(
                f"  - {item['historical_complaint_id']} | category={item['category']} | "
                f"priority={item['priority_label']} | eta={item['actual_resolution_days']} days | "
                f"similarity={item['similarity_score']:.3f}"
            )
            lines.append(f"    text: {item['normalized_text_en']}")
        return '\n'.join(lines)
