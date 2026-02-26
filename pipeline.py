"""
pipeline.py — text extraction, chunking, embedding, vector store ingestion

steps:
    1. extract text from all PDFs using pymupdf
    2. chunk into ~500 token pieces, respecting section breaks
    3. embed with multilingual-e5-base (handles nepali + english)
    4. store in chromadb

usage:
    python pipeline.py                  # process everything
    python pipeline.py --folder 11_laws_np   # one folder only
    python pipeline.py --extract-only   # stop after text extraction (useful for debugging)

requirements:
    pip install pymupdf sentence-transformers chromadb tqdm
"""

import os
import re
import json
import argparse
import hashlib
import logging
from pathlib import Path

import fitz  # pymupdf
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
TEXT_DIR = Path("data/processed/text")
CHUNKS_DIR = Path("data/processed/chunks")
CHROMA_DIR = Path("data/chromadb")

EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "civiclens_nepal"

# optional — only needed for OCR fallback on scanned PDFs
# set in .env file: POPPLER_PATH=C:\path\to\poppler\Library\bin
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

CHUNK_SIZE = 250      # target tokens per chunk (rough — we use word count as proxy)
CHUNK_OVERLAP = 50    # overlap between chunks so context isn't lost at boundaries
MIN_CHUNK_WORDS = 40   # chunks smaller than this get merged with neighbors
MAX_MERGE_COUNT = 4

# maps folder name → metadata category tag
FOLDER_CATEGORIES = {
    "01_constitution":     "constitution",
    "02_economic_survey":  "economic_survey",
    "03_budget_speech":    "budget_speech",
    "04_psc_syllabuses":   "psc_syllabus",
    "05_civil_service_acts": "civil_service",
    "06_bonus_resources":  "statistics",
    "07_key_laws":         "key_laws",
    "08_planning":         "planning",
    "09_npc_documents":    "npc",
    "10_laws_en":          "laws_en",
    "11_laws_np":          "laws_np",
}

# ── step 1: pdf text extraction ───────────────────────────────────────────────

def clean_text(text: str) -> str:
    text_no_dots    = re.sub(r'(?:[ \t]*\.[ \t]*){5,}', ' ', text)
    text_no_numbers = re.sub(r'^\s*\d+\s*$', '', text_no_dots, flags=re.MULTILINE)
    cleaned_text    = re.sub(r'^\s*\n', '', text_no_numbers, flags=re.MULTILINE)
    return cleaned_text


def is_garbage_text(text: str, source_path: Path = None) -> bool:
    if not text:
        return True

    # check 1: low ratio of clean characters (scanned/broken PDFs)
    total = len(text)
    clean = sum(1 for c in text if c.isalnum() or "\u0900" <= c <= "\u097f" or c == " ")
    if clean / total < 0.5:
        return True

    # check 2: nepali filename but zero devanagari in text (font-encoded e.g. Preeti)
    if source_path is not None:
        filename_is_nepali = any("\u0900" <= c <= "\u097f" for c in source_path.stem)
        text_has_devanagari = any("\u0900" <= c <= "\u097f" for c in text)
        if filename_is_nepali and not text_has_devanagari:
            return True

    return False


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Primary extraction using PyMuPDF — fast, works on text-layer PDFs."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        log.warning(f"  pymupdf failed: {pdf_path.name} — {e}")
        return ""


def extract_text_ocr(pdf_path: Path) -> str:
    """Fallback OCR extraction for scanned/garbage PDFs using tesseract."""
    if not POPPLER_PATH:
        log.warning(f"  OCR skipped: POPPLER_PATH not set in .env — {pdf_path.name}")
        return ""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        log.info(f"  falling back to OCR: {pdf_path.name}")
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        pages = []
        for img in images:
            text = pytesseract.image_to_string(img, lang="nep+eng")
            if text.strip():
                pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        log.warning(f"  OCR also failed: {pdf_path.name} — {e}")
        return ""


def extract_text(pdf_path: Path) -> tuple[str, str]:
    """
    Extract text from a PDF.
    Returns (text, method) where method is 'pymupdf' or 'ocr'.
    Falls back to OCR if pymupdf produces garbage.
    """
    text = extract_text_pymupdf(pdf_path)
    if is_garbage_text(text, source_path=pdf_path):
        text = extract_text_ocr(pdf_path)
        return text, "ocr"
    return text, "pymupdf"


def detect_language(text: str) -> str:
    # simple heuristic: if >20% of chars are devanagari, it's nepali
    if not text:
        return "unknown"
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    return "np" if devanagari / max(len(text), 1) > 0.2 else "en"


def run_extraction(folders=None):
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = []
    for folder in RAW_DIR.iterdir():
        if not folder.is_dir():
            continue
        if folders and folder.name not in folders:
            continue
        pdfs.extend(folder.glob("*.pdf"))

    log.info(f"extracting text from {len(pdfs)} PDFs...")
    skipped = extracted = failed = 0

    for pdf_path in tqdm(pdfs, desc="extracting"):
        out_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        if out_path.exists():
            skipped += 1
            continue

        log.info(f"  [{extracted + failed + 1}] {pdf_path.name}")
        text, method = extract_text(pdf_path)

        if not text.strip():
            failed += 1
            log.warning(f"  FAILED — empty result: {pdf_path.name}")
            continue

        text = clean_text(text)
        log.info(f"  method={method}  chars={len(text):,}  lang={detect_language(text)}")

        # save text with metadata header
        folder = pdf_path.parent.name
        lang = detect_language(text)
        meta = {
            "source_file":       pdf_path.name,
            "folder":            folder,
            "category":          FOLDER_CATEGORIES.get(folder, "other"),
            "language":          lang,
            "char_count":        len(text),
            "extraction_method": method,   # "pymupdf" or "ocr"
        }
        out_path.write_text(
            json.dumps(meta, ensure_ascii=False) + "\n---\n" + text,
            encoding="utf-8"
        )
        extracted += 1

    log.info(f"extraction done — extracted:{extracted}  skipped:{skipped}  failed:{failed}")
    return extracted + skipped  # total available


# ── step 2: chunking ──────────────────────────────────────────────────────────

# section heading patterns in both english and nepali
# Uselss the way i am doing right now, might change chuking start later for now, fuck it we ball
SECTION_PATTERNS = [
    re.compile(r"^\s*(?:section|article|chapter|part)\s+\d+", re.I | re.M),
    re.compile(r"^\s*(?:दफा|धारा|अनुसूची|भाग)\s*\d*", re.M),  # common nepali legal terms
]


# Also useless for now lol
def find_section_breaks(text: str) -> list[int]:
    breaks = set()
    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            breaks.add(m.start())
    return sorted(breaks)


def word_count(text: str) -> int:
    return len(text.split())

def chunk_text(text: str, meta: dict) -> list[dict]:
    words = text.split()
    chunks = []
    chunk_index = 0

    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]

        chunks.append({
            **meta,
            "chunk_index": chunk_index,
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words),
        })

        chunk_index += 1

        if end == len(words):
            break

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# Useless for now cuz i am doing a simple way
def merge_small_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        current = dict(chunks[i])
        merge_count = 1

        while (
            merge_count < MAX_MERGE_COUNT
            and i + merge_count < len(chunks)
            and (
                current["word_count"] < MIN_CHUNK_WORDS
                or chunks[i + merge_count]["word_count"] < MIN_CHUNK_WORDS
            )
        ):
            nxt = chunks[i + merge_count]
            current["text"] = current["text"].rstrip() + "\n" + nxt["text"].lstrip()
            current["word_count"] += nxt["word_count"]
            merge_count += 1

        merged.append(current)
        i += merge_count

    # keep chunk_index sequential
    for idx, chunk in enumerate(merged):
        chunk["chunk_index"] = idx

    return merged


def run_chunking(folders=None):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = list(TEXT_DIR.glob("*.txt"))
    if folders:
        # filter to only files from requested folders
        allowed_stems = set()
        for folder in RAW_DIR.iterdir():
            if folder.is_dir() and folder.name in folders:
                for p in folder.glob("*.pdf"):
                    allowed_stems.add(p.stem)
        txt_files = [f for f in txt_files if f.stem in allowed_stems]

    log.info(f"chunking {len(txt_files)} extracted texts...")
    total_chunks = 0

    for txt_path in tqdm(txt_files, desc="chunking"):
        out_path = CHUNKS_DIR / f"{txt_path.stem}.jsonl"
        if out_path.exists():
            continue

        content = txt_path.read_text(encoding="utf-8")
        if "\n---\n" not in content:
            continue

        meta_str, text = content.split("\n---\n", 1)
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            continue

        chunks = chunk_text(text, meta)
        if not chunks:
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        total_chunks += len(chunks)

    log.info(f"chunking done — {total_chunks} chunks across {len(txt_files)} docs")
    return total_chunks


# ── step 3: embedding + chromadb ingestion ────────────────────────────────────

def make_chunk_id(source_file: str, chunk_index: int) -> str:
    raw = f"{source_file}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_chunk_files(folders=None) -> list[Path]:
    chunk_files = list(CHUNKS_DIR.glob("*.jsonl"))
    if not folders:
        return chunk_files
    allowed_stems = set()
    for folder in RAW_DIR.iterdir():
        if folder.is_dir() and folder.name in folders:
            for p in folder.glob("*.pdf"):
                allowed_stems.add(p.stem)
    return [f for f in chunk_files if f.stem in allowed_stems]


def load_pending_chunks(chunk_files: list[Path], existing_ids: set) -> list[dict]:
    pending = []
    for chunk_file in chunk_files:
        with open(chunk_file, encoding="utf-8") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = make_chunk_id(chunk["source_file"], chunk["chunk_index"])
                if chunk_id not in existing_ids:
                    chunk["_id"] = chunk_id
                    pending.append(chunk)
    return pending


def embed_and_ingest(pending: list[dict], collection, model, batch_size: int):
    texts_to_embed = [f"passage: {c['text']}" for c in pending]

    for i in tqdm(range(0, len(pending), batch_size), desc="embedding"):
        batch_chunks = pending[i:i + batch_size]
        batch_texts  = texts_to_embed[i:i + batch_size]

        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=[c["_id"] for c in batch_chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[{
                "source_file": c["source_file"],
                "folder":      c["folder"],
                "category":    c["category"],
                "language":    c["language"],
                "chunk_index": c["chunk_index"],
                "word_count":  c["word_count"],
            } for c in batch_chunks],
        )


def run_embedding(batch_size=64, folders=None):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    existing_ids = set(collection.get(include=[])["ids"])
    log.info(f"chromadb has {len(existing_ids)} existing chunks")

    chunk_files = get_chunk_files(folders)
    pending = load_pending_chunks(chunk_files, existing_ids)

    log.info(f"{len(pending)} chunks to embed and ingest")
    if not pending:
        log.info("nothing to do — all chunks already in chromadb")
        return

    embed_and_ingest(pending, collection, model, batch_size)

    log.info(f"done — {len(pending)} chunks ingested into chromadb")
    log.info(f"total collection size: {collection.count()} chunks")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="process only this folder (e.g. 11_laws_np)")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--chunk-only", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    folders = {args.folder} if args.folder else None

    if args.embed_only:
        run_embedding(batch_size=args.batch_size, folders=folders)
        return
    if args.chunk_only:
        run_chunking(folders=folders)
        return

    run_extraction(folders=folders)
    if args.extract_only:
        return

    run_chunking(folders=folders)
    run_embedding(batch_size=args.batch_size, folders=folders)


if __name__ == "__main__":
    main()