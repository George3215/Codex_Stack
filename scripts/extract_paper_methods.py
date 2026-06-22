from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


KEYWORDS = re.compile(
    r"(abstract|algorithm|heuristic|sequence|stabil|reward|reinforcement|"
    r"q-learning|dqn|pose|target|planning|candidate|search|simulation|"
    r"physics|dry stone|wall|stack)",
    re.IGNORECASE,
)


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


def normalized_page_text(page: object) -> str:
    try:
        text = page.extract_text() or ""
    except Exception:
        text = ""
    return re.sub(r"\s+", " ", text)


def snippets(full_text: str, limit: int) -> list[str]:
    output: list[str] = []
    prefixes: set[str] = set()
    for match in KEYWORDS.finditer(full_text):
        start = max(0, match.start() - 260)
        end = min(len(full_text), match.end() + 540)
        snippet = full_text[start:end].strip()
        key = snippet[:120]
        if key in prefixes:
            continue
        prefixes.add(key)
        output.append(snippet)
        if len(output) >= limit:
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=Path, default=Path(r"D:\MoonStack\Asset\Papers"))
    parser.add_argument("--snippets", type=int, default=10)
    args = parser.parse_args()

    for path in unique_pdfs(args.papers):
        print("\n" + "=" * 100)
        print(path.name)
        reader = PdfReader(str(path))
        page_texts = [normalized_page_text(page) for page in reader.pages]
        full_text = " ".join(page_texts)
        print(f"pages: {len(reader.pages)}")
        print(f"text_chars: {len(full_text)}")
        if full_text:
            print("first_700:")
            print(full_text[:700])
        for idx, snippet in enumerate(snippets(full_text, args.snippets), 1):
            print(f"-- hit {idx} --")
            print(snippet[:900])


if __name__ == "__main__":
    main()
