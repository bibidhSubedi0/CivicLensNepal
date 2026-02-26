import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    extract_text,
    detect_language,
    is_garbage_text,
    is_preeti,
    clean_text,
)


def sanity_check(pdf_path: Path):
    print(f"\n{'='*60}")
    print(f"FILE:  {pdf_path.name}")
    print(f"SIZE:  {pdf_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")

    # --- raw pymupdf pass ---
    import fitz
    doc = fitz.open(pdf_path)
    raw_pages = [page.get_text("text") for page in doc]
    doc.close()
    raw_text = "\n".join(p for p in raw_pages if p.strip())

    print(f"[raw pymupdf]")
    print(f"  pages extracted : {len(raw_pages)}")
    print(f"  chars           : {len(raw_text):,}")
    print(f"  is_garbage      : {is_garbage_text(raw_text)}")
    print(f"  is_preeti       : {is_preeti(raw_text)}")
    print(f"  sample (100c)   : {repr(raw_text[:100])}\n")

    # --- full extraction with fallback ---
    print(f"[extract_text()]")
    text, method = extract_text(pdf_path)
    print(f"  method used     : {method}")

    if not text.strip():
        print("  RESULT: EMPTY — extraction failed completely")
        return

    text = clean_text(text)
    lang = detect_language(text)

    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    latin      = sum(1 for c in text if c.isascii() and c.isalpha())

    print(f"  chars (cleaned) : {len(text):,}")
    print(f"  words           : {len(text.split()):,}")
    print(f"  language        : {lang}")
    print(f"  devanagari %    : {devanagari / max(len(text), 1) * 100:.1f}%")
    print(f"  latin %         : {latin / max(len(text), 1) * 100:.1f}%")
    print(f"\n[sample — first 500 chars]\n")
    print(text[:500])
    print(f"\n[sample — middle 500 chars]\n")
    mid = len(text) // 2
    print(text[mid:mid+500])
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python test.py <path/to/file.pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"error: file not found — {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"error: expected a .pdf file")
        sys.exit(1)

    sanity_check(pdf_path)