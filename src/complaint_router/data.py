from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

TEXT_COL = 'normalized_text_en'
ORIGINAL_TEXT_COL = 'original_text'


@dataclass
class Datasets:
    complaints: pd.DataFrame
    officers: pd.DataFrame
    historical: pd.DataFrame


def load_datasets(
    complaints_path: Path,
    officers_path: Path,
    historical_path: Path,
) -> Datasets:
    complaints = pd.read_csv(complaints_path)
    officers = pd.read_csv(officers_path)
    historical = pd.read_csv(historical_path)

    for df in (complaints, historical):
        for col in [TEXT_COL, ORIGINAL_TEXT_COL, 'city', 'ward', 'category', 'subcategory', 'language']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)

    officers['languages_supported'] = officers['languages_supported'].fillna('en').astype(str)
    officers['specialization_category'] = officers['specialization_category'].fillna('').astype(str)
    officers['city'] = officers['city'].fillna('').astype(str)
    officers['primary_ward'] = officers['primary_ward'].fillna('').astype(str)

    return Datasets(complaints=complaints, officers=officers, historical=historical)
