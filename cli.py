from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.complaint_router.config import MODELS_DIR
from src.complaint_router.pipeline import ComplaintInput, ComplaintRoutingPipeline


AUDIO_VIDEO_NOTE = (
    'For audio/video in this offline baseline, pass a transcription via --text or --transcript-file. '
    'This keeps the CLI API-ready while avoiding external services.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Complaint auto-routing CLI')
    parser.add_argument('--models-dir', type=Path, default=MODELS_DIR)
    parser.add_argument('--text', type=str, default='')
    parser.add_argument('--transcript-file', type=Path)
    parser.add_argument('--input-type', type=str, default='text', choices=['text', 'audio', 'video'])
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--subcategory', type=str, required=True)
    parser.add_argument('--city', type=str, required=True)
    parser.add_argument('--ward', type=str, required=True)
    parser.add_argument('--attachments-count', type=int, default=0)
    parser.add_argument('--citizen-sentiment-score', type=float, default=0.0)
    parser.add_argument('--json', action='store_true', help='Print raw JSON output')
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
    if args.input_type in {'audio', 'video'}:
        print(f'[info] {AUDIO_VIDEO_NOTE}')

    complaint = ComplaintInput(
        text=text,
        category=args.category,
        subcategory=args.subcategory,
        city=args.city,
        ward=args.ward,
        language=args.language,
        input_type=args.input_type,
        attachments_count=args.attachments_count,
        citizen_sentiment_score=args.citizen_sentiment_score,
    )

    pipeline = ComplaintRoutingPipeline(args.models_dir)
    result = pipeline.predict(complaint)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(ComplaintRoutingPipeline.pretty_print(result))


if __name__ == '__main__':
    main()
