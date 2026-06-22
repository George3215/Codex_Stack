from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def unique_pdfs(root: Path) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.pdf")):
        key = path.name.replace(" (1).pdf", ".pdf")
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def page_texts(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    output: list[str] = []
    for page in reader.pages:
        try:
            output.append(clean(page.extract_text() or ""))
        except Exception:
            output.append("")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=Path, default=Path(r"D:\MoonStack\Asset\Papers"))
    parser.add_argument("--file", default="")
    parser.add_argument("--terms", required=True)
    parser.add_argument("--chars", type=int, default=850)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    terms = [term.strip() for term in args.terms.split(",") if term.strip()]
    for path in unique_pdfs(args.papers):
        if args.file and args.file.lower() not in path.name.lower():
            continue
        texts = page_texts(path)
        print("\n" + "=" * 100)
        print(path.name)
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            hits = 0
            print(f"\nTERM: {term}")
            for page_no, text in enumerate(texts, start=1):
                for match in pattern.finditer(text):
                    start = max(0, match.start() - args.chars // 2)
                    end = min(len(text), match.end() + args.chars // 2)
                    print(f"-- page {page_no} --")
                    print(text[start:end])
                    hits += 1
                    if hits >= args.limit:
                        break
                if hits >= args.limit:
                    break
            if hits == 0:
                print("(no hits)")


if __name__ == "__main__":
    main()
