from __future__ import annotations

import json
from src.complaint_router.config import EVAL_PATH

def main() -> None:
    data = json.loads(EVAL_PATH.read_text())
    print('=== Evaluation Summary ===')
    for key, value in data.items():
        print(f'{key}: {value:.4f}')

if __name__ == '__main__':
    main()
