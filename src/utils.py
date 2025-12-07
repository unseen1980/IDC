import os
import json
from pathlib import Path
from typing import Iterable, Dict, List


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text_files(input_dir: str | Path, glob: str = "*.txt") -> List[Dict]:
    input_dir = Path(input_dir)
    docs = []
    for p in sorted(input_dir.rglob(glob)):
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = p.read_text(encoding="latin-1")
            doc_id = p.stem
            docs.append({"doc_id": doc_id, "path": str(p), "text": text})
    return docs
