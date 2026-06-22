from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


KEYWORDS = (
    "dry stone",
    "dry-stack",
    "dry stacking",
    "wall",
    "stone",
    "planning",
    "sequence",
    "stability",
    "pose",
    "target",
    "structure",
    "placement",
    "height map",
    "reinforcement",
    "support",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract short method snippets from dry-stacking papers.")
    parser.add_argument("--papers-dir", type=Path, default=Path("D:/MoonStack/Asset/Papers"))
    parser.add_argument("--max-snippets", type=int, default=16)
    args = parser.parse_args()

    paths = sorted(path for path in args.papers_dir.glob("*.pdf") if "(1)" not in path.name)
    for path in paths:
        if not any(token in path.name.lower() for token in ("dry", "stack", "stone", "walls")):
            continue
        print(f"\n=== {path.name} ===")
        reader = PdfReader(str(path))
        print(f"pages: {len(reader.pages)}")
        emitted = 0
        for page_index, page in enumerate(reader.pages):
            if emitted >= args.max_snippets:
                break
            text = " ".join((page.extract_text() or "").split())
            if not text:
                continue
            lowered = text.lower()
            hits = [keyword for keyword in KEYWORDS if keyword in lowered]
            if not hits:
                continue
            for keyword in hits[:2]:
                if emitted >= args.max_snippets:
                    break
                match = re.search(re.escape(keyword), lowered)
                if not match:
                    continue
                start = max(0, match.start() - 260)
                end = min(len(text), match.end() + 420)
                snippet = text[start:end]
                print(f"[p{page_index + 1:02d}] keyword={keyword}")
                print(snippet)
                emitted += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
