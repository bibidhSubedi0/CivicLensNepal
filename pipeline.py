import re
import json
import argparse
import hashlib
import logging
from pathlib import Path

import fitz
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


HUNK_SIZE = 500      # target tokens per chunk (rough, we use word count as approximation)
CHUNK_OVERLAP = 50    # overlap between chunks so context isn't lost at boundaries

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
    "10_laws_en":          "laws_en", # dont have rn, will extract later
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


