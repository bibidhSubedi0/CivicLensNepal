import os
import re
import io
import sys
import json
import argparse
import hashlib
import logging
import traceback
import time
from pathlib import Path
from datetime import datetime

import fitz
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

RAW_DIR    = Path("data/raw")
TEXT_DIR   = Path("data/processed/text")
CHUNKS_DIR = Path("data/processed/chunks")
CHROMA_DIR = Path("data/chromadb")

EMBED_MODEL     = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "civiclens_nepal"

NEP_TTF2UTF_DIR = Path("nep-ttf2utf")
POPPLER_PATH    = os.getenv("POPPLER_PATH", None)
OCR_TIMEOUT     = 600  # seconds — skip file if OCR takes longer than this

CHUNK_SIZE      = 250
CHUNK_OVERLAP   = 50
MIN_CHUNK_WORDS = 40
MAX_MERGE_COUNT = 4

_PREETI_SUSPECT = frozenset(r'[]{}\|^~`<>')

FOLDER_CATEGORIES = {
    "01_constitution":       "constitution",
    "02_economic_survey":    "economic_survey",
    "03_budget_speech":      "budget_speech",
    "04_psc_syllabuses":     "psc_syllabus",
    "05_civil_service_acts": "civil_service",
    "06_bonus_resources":    "statistics",
    "07_key_laws":           "key_laws",
    "08_planning":           "planning",
    "09_npc_documents":      "npc",
    "10_laws_en":            "laws_en",
    "11_laws_np":            "laws_np",
}

# ── device detection ──────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) — running on CUDA")
else:
    log.info("No GPU detected — running on CPU")


# ── Preeti converter setup ────────────────────────────────────────────────────

def _load_preeti_converter():
    if not NEP_TTF2UTF_DIR.exists():
        log.warning("nep-ttf2utf not found — Preeti PDFs will fall back to OCR.")
        return None
    try:
        sys.path.insert(0, str(NEP_TTF2UTF_DIR))
        from ttf2utf import converter, rules_loader
        rules_path = str(NEP_TTF2UTF_DIR / "rules") + os.sep
        all_rules  = rules_loader.load_rules(rules_path)
        rule       = all_rules["preeti"]

        def preeti_to_unicode(text: str) -> str:
            in_file  = io.StringIO(text)
            out_file = io.StringIO()
            converter.convert(rule, in_file, out_file)
            return out_file.getvalue()

        log.info("Preeti converter loaded")
        return preeti_to_unicode

    except Exception as e:
        log.warning(f"Failed to load Preeti converter: {e} — will fall back to OCR")
        return None


_preeti_to_unicode = _load_preeti_converter()


# ── easyocr setup ─────────────────────────────────────────────────────────────

def _load_easyocr():
    try:
        import easyocr
        use_gpu = (DEVICE == "cuda")
        reader  = easyocr.Reader(["ne", "en"], gpu=use_gpu, verbose=False)
        log.info(f"easyocr loaded (gpu={use_gpu})")
        return reader
    except ImportError:
        log.info("easyocr not installed — OCR will use tesseract instead")
        return None
    except Exception as e:
        log.warning(f"easyocr failed to load: {e} — will use tesseract")
        return None


_easyocr_reader = _load_easyocr()


# ── text extraction ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'(?:[ \t]*\.[ \t]*){5,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)
    return text


def is_garbage_text(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return True

    control_chars = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if control_chars / len(text) > 0.1:
        return True

    words = [w for w in text.split() if len(w) > 1]
    if len(words) < 30:
        return True

    total      = len(text)
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    latin      = sum(1 for c in text if c.isascii() and c.isalpha())
    digits     = sum(1 for c in text if c.isdigit())
    whitespace = sum(1 for c in text if c in " \n\t\r")
    suspect    = sum(1 for c in text if c in _PREETI_SUSPECT)
    meaningful = devanagari + latin + digits + whitespace

    if meaningful / total < 0.5:
        return True
    if devanagari / total > 0.15:
        return False
    if suspect / total > 0.04:
        return True

    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2.0 or avg_len > 25.0:
        return True

    return False


def is_preeti(text: str) -> bool:
    if not text or len(text) < 50:
        return False
    total      = len(text)
    suspect    = sum(1 for c in text if c in _PREETI_SUSPECT)
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    return suspect / total > 0.04 and devanagari / total < 0.05


def extract_text_pymupdf(pdf_path: Path) -> str:
    try:
        doc   = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(p for p in pages if p.strip())
    except Exception as e:
        log.warning(f"  pymupdf failed: {pdf_path.name} — {e}")
        return ""


def extract_text_preeti(raw_text: str) -> str:
    if _preeti_to_unicode is None:
        return ""
    try:
        return _preeti_to_unicode(raw_text)
    except Exception as e:
        log.warning(f"  Preeti conversion failed — {e}")
        return ""


def extract_text_ocr(pdf_path: Path) -> str:
    try:
        doc        = fitz.open(pdf_path)
        page_count = doc.page_count
        pages      = []

        if _easyocr_reader is not None:
            log.info(f"  OCR via easyocr ({'GPU' if DEVICE == 'cuda' else 'CPU'}): {pdf_path.name}")
            import numpy as np

            ocr_start = time.time()
            for page_num in range(page_count):
                if time.time() - ocr_start > OCR_TIMEOUT:
                    log.warning(f"  OCR timeout ({OCR_TIMEOUT}s) on {pdf_path.name} at page {page_num + 1} — skipping rest")
                    break
                try:
                    page = doc[page_num]
                    mat  = fitz.Matrix(150 / 72, 150 / 72)
                    pix  = page.get_pixmap(matrix=mat)

                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4:
                        img = img[:, :, :3]

                    results = _easyocr_reader.readtext(img, detail=0, paragraph=True)
                    text    = "\n".join(results)
                    if text.strip():
                        pages.append(text)

                except MemoryError:
                    log.warning(f"  OOM on page {page_num + 1}, skipping")
                except Exception as e:
                    log.warning(f"  easyocr failed on page {page_num + 1}: {e}")

            doc.close()
            return "\n".join(pages)

        # tesseract fallback
        doc.close()
        if not POPPLER_PATH:
            log.warning(f"  OCR skipped: neither easyocr nor POPPLER_PATH available — {pdf_path.name}")
            return ""

        from pdf2image import convert_from_path
        import pytesseract

        log.info(f"  OCR via tesseract (CPU): {pdf_path.name}")
        for page_num in range(1, page_count + 1):
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=150,
                    poppler_path=POPPLER_PATH,
                    first_page=page_num,
                    last_page=page_num,
                )
                text = pytesseract.image_to_string(images[0], lang="nep+eng")
                if text.strip():
                    pages.append(text)
            except MemoryError:
                log.warning(f"  OOM on page {page_num}, skipping")
            except Exception as e:
                log.warning(f"  tesseract failed on page {page_num}: {e}")

        return "\n".join(pages)

    except Exception as e:
        log.warning(f"  OCR completely failed: {pdf_path.name} — {e}")
        return ""


def extract_text(pdf_path: Path, skip_ocr: bool = False) -> tuple[str, str]:
    text = extract_text_pymupdf(pdf_path)

    if not is_garbage_text(text):
        return text, "pymupdf"

    if is_preeti(text):
        log.info(f"  Preeti font detected: {pdf_path.name}")
        converted = extract_text_preeti(text)
        if converted.strip() and not is_garbage_text(converted):
            return converted, "preeti"
        log.warning(f"  Preeti conversion produced garbage, falling back to OCR: {pdf_path.name}")

    if skip_ocr:
        log.info(f"  skipping OCR (--skip-ocr): {pdf_path.name}")
        return "", "skipped"

    text = extract_text_ocr(pdf_path)
    return text, "ocr"


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    return "np" if devanagari / max(len(text), 1) > 0.2 else "en"


def run_extraction(folders=None, skip_ocr: bool = False):
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = []
    for folder in RAW_DIR.iterdir():
        if not folder.is_dir():
            continue
        if folders and folder.name not in folders:
            continue
        pdfs.extend(folder.glob("*.pdf"))

    log.info(f"extracting text from {len(pdfs)} PDFs...")
    skipped = extracted = failed = crashed = 0
    failed_files = []

    for pdf_path in tqdm(pdfs, desc="extracting"):
        out_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        if out_path.exists():
            skipped += 1
            continue

        log.info(f"  [{extracted + failed + crashed + 1}] {pdf_path.name}")

        try:
            text, method = extract_text(pdf_path, skip_ocr=skip_ocr)

            if not text.strip():
                failed += 1
                failed_files.append((pdf_path.name, "empty result"))
                log.warning(f"  FAILED — empty result: {pdf_path.name}")
                continue

            text = clean_text(text)
            lang = detect_language(text)
            log.info(f"  method={method}  chars={len(text):,}  lang={lang}")

            folder = pdf_path.parent.name
            meta   = {
                "source_file":       pdf_path.name,
                "folder":            folder,
                "category":          FOLDER_CATEGORIES.get(folder, "other"),
                "language":          lang,
                "char_count":        len(text),
                "extraction_method": method,
            }
            out_path.write_text(
                json.dumps(meta, ensure_ascii=False) + "\n---\n" + text,
                encoding="utf-8-sig"
            )
            extracted += 1

        except Exception as e:
            crashed += 1
            failed_files.append((pdf_path.name, str(e)))
            log.error(f"  CRASHED on {pdf_path.name}: {e}")
            log.debug(traceback.format_exc())
            # never stop — move on to next file
            continue

    # ── final report ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"EXTRACTION COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  extracted : {extracted}")
    log.info(f"  skipped   : {skipped}")
    log.info(f"  failed    : {failed}")
    log.info(f"  crashed   : {crashed}")
    if failed_files:
        log.info("  problem files:")
        for name, reason in failed_files:
            log.info(f"    {name} — {reason}")
    log.info("=" * 60)

    return extracted + skipped


# ── chunking ──────────────────────────────────────────────────────────────────

SECTION_PATTERNS = [
    re.compile(r"^\s*(?:section|article|chapter|part)\s+\d+", re.I | re.M),
    re.compile(r"^\s*(?:दफा|धारा|अनुसूची|भाग)\s*\d*", re.M),
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
    words       = text.split()
    chunks      = []
    chunk_index = 0
    start       = 0

    while start < len(words):
        end         = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]
        chunks.append({
            **meta,
            "chunk_index": chunk_index,
            "text":        " ".join(chunk_words),
            "word_count":  len(chunk_words),
        })
        chunk_index += 1
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def merge_small_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        current     = dict(chunks[i])
        merge_count = 1
        while (
            merge_count < MAX_MERGE_COUNT
            and i + merge_count < len(chunks)
            and (
                current["word_count"] < MIN_CHUNK_WORDS
                or chunks[i + merge_count]["word_count"] < MIN_CHUNK_WORDS
            )
        ):
            nxt             = chunks[i + merge_count]
            current["text"] = current["text"].rstrip() + "\n" + nxt["text"].lstrip()
            current["word_count"] += nxt["word_count"]
            merge_count += 1
        merged.append(current)
        i += merge_count

    for idx, chunk in enumerate(merged):
        chunk["chunk_index"] = idx

    return merged


def run_chunking(folders=None):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = list(TEXT_DIR.glob("*.txt"))

    if folders:
        allowed_stems = set()
        for folder in RAW_DIR.iterdir():
            if folder.is_dir() and folder.name in folders:
                for p in folder.glob("*.pdf"):
                    allowed_stems.add(p.stem)
        txt_files = [f for f in txt_files if f.stem in allowed_stems]

    log.info(f"chunking {len(txt_files)} extracted texts...")
    total_chunks = 0
    failed = 0

    for txt_path in tqdm(txt_files, desc="chunking"):
        out_path = CHUNKS_DIR / f"{txt_path.stem}.jsonl"
        if out_path.exists():
            continue

        try:
            content = txt_path.read_text(encoding="utf-8-sig")
            if "\n---\n" not in content:
                continue

            meta_str, text = content.split("\n---\n", 1)
            meta   = json.loads(meta_str)
            chunks = chunk_text(text, meta)
            if not chunks:
                continue

            with open(out_path, "w", encoding="utf-8-sig") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)

        except Exception as e:
            failed += 1
            log.error(f"  chunking failed for {txt_path.name}: {e}")
            continue

    log.info(f"chunking done — {total_chunks} chunks across {len(txt_files)} docs  (failed: {failed})")
    return total_chunks


# ── embedding + chromadb ingestion ────────────────────────────────────────────

def make_chunk_id(source_file: str, chunk_index: int) -> str:
    return hashlib.md5(f"{source_file}_{chunk_index}".encode()).hexdigest()


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
        try:
            with open(chunk_file, encoding="utf-8-sig") as f:
                for line in f:
                    try:
                        chunk    = json.loads(line)
                        chunk_id = make_chunk_id(chunk["source_file"], chunk["chunk_index"])
                        if chunk_id not in existing_ids:
                            chunk["_id"] = chunk_id
                            pending.append(chunk)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.error(f"  failed to read chunk file {chunk_file.name}: {e}")
            continue
    return pending


def embed_and_ingest(pending: list[dict], collection, model, batch_size: int):
    texts_to_embed = [f"passage: {c['text']}" for c in pending]

    log.info(f"encoding {len(texts_to_embed)} chunks on {DEVICE.upper()}...")
    all_embeddings = model.encode(
        texts_to_embed,
        batch_size           = batch_size,
        normalize_embeddings = True,
        show_progress_bar    = True,
        convert_to_numpy     = True,
    )

    chroma_batch = 512
    log.info("ingesting into chromadb...")
    for i in tqdm(range(0, len(pending), chroma_batch), desc="ingesting"):
        try:
            batch_chunks     = pending[i:i + chroma_batch]
            batch_embeddings = all_embeddings[i:i + chroma_batch].tolist()
            collection.add(
                ids        = [c["_id"] for c in batch_chunks],
                embeddings = batch_embeddings,
                documents  = [c["text"] for c in batch_chunks],
                metadatas  = [{
                    "source_file":       c["source_file"],
                    "folder":            c["folder"],
                    "category":          c["category"],
                    "language":          c["language"],
                    "chunk_index":       c["chunk_index"],
                    "word_count":        c["word_count"],
                    "extraction_method": c.get("extraction_method", "unknown"),
                } for c in batch_chunks],
            )
        except Exception as e:
            log.error(f"  chromadb batch {i} failed: {e} — continuing")
            continue


def run_embedding(batch_size=64, folders=None):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"loading embedding model: {EMBED_MODEL} on {DEVICE.upper()}")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1e9
        log.info(f"model loaded — VRAM in use: {allocated:.2f} GB")

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )

    existing_ids = set(collection.get(include=[])["ids"])
    log.info(f"chromadb has {len(existing_ids)} existing chunks")

    chunk_files = get_chunk_files(folders)
    pending     = load_pending_chunks(chunk_files, existing_ids)

    log.info(f"{len(pending)} chunks to embed and ingest")
    if not pending:
        log.info("nothing to do — all chunks already in chromadb")
        return

    embed_and_ingest(pending, collection, model, batch_size)

    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        log.info(f"peak VRAM used during embedding: {peak:.2f} GB")

    log.info("=" * 60)
    log.info(f"EMBEDDING COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  total collection size: {collection.count()} chunks")
    log.info("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",       help="process only this folder")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--chunk-only",   action="store_true")
    parser.add_argument("--embed-only",   action="store_true")
    parser.add_argument("--skip-ocr",     action="store_true")
    parser.add_argument("--batch-size",   type=int, default=64)
    args = parser.parse_args()

    folders = {args.folder} if args.folder else None

    log.info(f"pipeline started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.embed_only:
        run_embedding(batch_size=args.batch_size, folders=folders)
        return
    if args.chunk_only:
        run_chunking(folders=folders)
        return

    run_extraction(folders=folders, skip_ocr=args.skip_ocr)
    if args.extract_only:
        return

    run_chunking(folders=folders)
    run_embedding(batch_size=args.batch_size, folders=folders)

    log.info("pipeline fully complete — shutting down...")
    os.system("shutdown /s /t 30")  # 30 second delay so logs flush


if __name__ == "__main__":
    main()
