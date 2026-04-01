from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.complaint_router.config import MODELS_DIR
from src.complaint_router.pipeline import ComplaintInput, ComplaintRoutingPipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Complaint auto-routing CLI')
    parser.add_argument('--models-dir', type=Path, default=MODELS_DIR)
    parser.add_argument('--text', type=str, default='')
    parser.add_argument('--transcript-file', type=Path)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--subcategory', type=str, required=True)
    parser.add_argument('--city', type=str, required=True)
    parser.add_argument('--ward', type=str, required=True)
    parser.add_argument('--attachments-count', type=int, default=0)
    parser.add_argument('--citizen-sentiment-score', type=float, default=0.0)
    return parser.parse_args()

def read_text(args: argparse.Namespace) -> str:
    if args.text.strip():
        return args.text.strip()
    if args.transcript_file:
        return args.transcript_file.read_text(encoding='utf-8').strip()
    raise SystemExit('Provide complaint content via --text or --transcript-file.')

def main() -> None:
    args = parse_args()
    text = read_text(args)

    complaint = ComplaintInput(
        text=text,
        category=args.category,
        subcategory=args.subcategory,
        city=args.city,
        ward=args.ward,
        language=args.language,
        attachments_count=args.attachments_count,
        citizen_sentiment_score=args.citizen_sentiment_score,
    )
    pipeline = ComplaintRoutingPipeline(args.models_dir)
    result = pipeline.predict(complaint)
    print(ComplaintRoutingPipeline.pretty_print(result))


if __name__ == '__main__':
    main()
