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

import re
import json
import argparse
import hashlib
import logging
from pathlib import Path

import fitz  # pymupdf
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
TEXT_DIR = Path("data/processed/text")
CHUNKS_DIR = Path("data/processed/chunks")
CHROMA_DIR = Path("data/chromadb")

EMBED_MODEL = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "civiclens_nepal"

CHUNK_SIZE = 500      # target tokens per chunk (rough — we use word count as proxy)
CHUNK_OVERLAP = 50    # overlap between chunks so context isn't lost at boundaries

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

def extract_text(pdf_path: Path) -> str:
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
        log.warning(f"extraction failed: {pdf_path.name} — {e}")
        return ""


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

        text = extract_text(pdf_path)
        if not text.strip():
            failed += 1
            log.warning(f"  empty: {pdf_path.name}")
            continue

        # save text with metadata header
        folder = pdf_path.parent.name
        lang = detect_language(text)
        meta = {
            "source_file": pdf_path.name,
            "folder": folder,
            "category": FOLDER_CATEGORIES.get(folder, "other"),
            "language": lang,
            "char_count": len(text),
        }
        out_path.write_text(
            json.dumps(meta) + "\n---\n" + text,
            encoding="utf-8"
        )
        extracted += 1

    log.info(f"extraction done — extracted:{extracted}  skipped:{skipped}  failed:{failed}")
    return extracted + skipped  # total available


# ── step 2: chunking ──────────────────────────────────────────────────────────

# section heading patterns in both english and nepali
SECTION_PATTERNS = [
    re.compile(r"^\s*(?:section|article|chapter|part)\s+\d+", re.I | re.M),
    re.compile(r"^\s*\d+[\.\)]\s+\S", re.M),           # 1. or 1) followed by text
    re.compile(r"^\s*[१२३४५६७८९०]+[\.\)]\s+\S", re.M), # nepali numerals
    re.compile(r"^\s*(?:दफा|धारा|अनुसूची|भाग)\s*\d*", re.M),  # common nepali legal terms
]


def find_section_breaks(text: str) -> list[int]:
    breaks = set()
    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            breaks.add(m.start())
    return sorted(breaks)


def word_count(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, meta: dict) -> list[dict]:
    chunks = []
    breaks = find_section_breaks(text)

    # split into sections first, then chunk each section if too long
    if breaks:
        sections = []
        prev = 0
        for b in breaks:
            if b > prev:
                sections.append(text[prev:b])
            prev = b
        sections.append(text[prev:])
    else:
        sections = [text]

    chunk_index = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
        words = section.split()

        if len(words) <= CHUNK_SIZE:
            # section fits in one chunk
            chunks.append({
                **meta,
                "chunk_index": chunk_index,
                "text": section,
                "word_count": len(words),
            })
            chunk_index += 1
        else:
            # slide a window over the section
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
                start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


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

    chunk_files = list(CHUNKS_DIR.glob("*.jsonl"))
    if folders:
        allowed_stems = set()
        for folder in RAW_DIR.iterdir():
            if folder.is_dir() and folder.name in folders:
                for p in folder.glob("*.pdf"):
                    allowed_stems.add(p.stem)
        chunk_files = [f for f in chunk_files if f.stem in allowed_stems]

    # load all chunks not yet in the db
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

    log.info(f"{len(pending)} chunks to embed and ingest")
    if not pending:
        log.info("nothing to do — all chunks already in chromadb")
        return

    # multilingual-e5 works best with a "query: " or "passage: " prefix
    texts_to_embed = [f"passage: {c['text']}" for c in pending]

    for i in tqdm(range(0, len(pending), batch_size), desc="embedding"):
        batch_chunks = pending[i:i + batch_size]
        batch_texts = texts_to_embed[i:i + batch_size]

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