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

# paths
RAW_DIR    = Path("data/raw")
TEXT_DIR   = Path("data/processed/text")
CHUNKS_DIR = Path("data/processed/chunks")
CHROMA_DIR = Path("data/chromadb")

EMBED_MODEL     = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "civiclens_nepal"

NEP_TTF2UTF_DIR = Path("nep-ttf2utf")
POPPLER_PATH    = os.getenv("POPPLER_PATH", None)
OCR_TIMEOUT     = 600  # some PDFs are 300+ pages and i have 8 yo gtx1650

# chunking params = 250/50 overlap felt right after testing on a few law docs
CHUNK_SIZE      = 250
CHUNK_OVERLAP   = 50
MIN_CHUNK_WORDS = 40
MAX_MERGE_COUNT = 4

# these characters show up a lot in Preeti-encoded PDFs (legacy Nepali font)
# if you see a lot of these, the PDF is probably not unicode
_PREETI_SUSPECT = frozenset(r'[]{}\\|^~`<>')

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM), running on CUDA")
else:
    log.info("No GPU, running on CPU (embedding will be slow)")


# ── Preeti font converter ─────────────────────────────────────────────────────
# A lot of older Nepali govt documents use Preeti, a legacy ASCII-mapped font.
# They look like Devanagari on screen but the underlying bytes are garbage unicode.
# nep-ttf2utf handles the conversion, clone it from github.com/sapradhan/nep-ttf2utf
# A HUGEEE THANK YOU TO MR SAPRADHAN!

def _load_preeti_converter():
    if not NEP_TTF2UTF_DIR.exists():
        log.warning("nep-ttf2utf not found, Preeti PDFs will fall back to OCR")
        return None
    try:
        sys.path.insert(0, str(NEP_TTF2UTF_DIR))
        from ttf2utf import converter, rules_loader  # type: ignore

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
        log.warning(f"Preeti converter failed to load: {e}")
        return None


_preeti_to_unicode = _load_preeti_converter()


# ── OCR setup ─────────────────────────────────────────────────────────────────
# easyocr > tesseract for Nepali. tesseract's nepali model is apparently pretty bad also no gpu support.
# friendship with tesseract is over. easyocr is my new best friend.

def _load_easyocr():
    try:
        import easyocr
        reader = easyocr.Reader(["ne", "en"], gpu=(DEVICE == "cuda"), verbose=False)
        log.info(f"easyocr loaded (gpu={DEVICE == 'cuda'})")
        return reader
    except ImportError:
        log.info("easyocr not installed, falling back to tesseract")
        return None
    except Exception as e:
        log.warning(f"easyocr failed to init: {e}")
        return None


_easyocr_reader = _load_easyocr()


# ── text extraction ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    # strip table-of-contents dot leaders like "Chapter 3 ............... 14"
    text = re.sub(r'(?:[ \t]*\.[ \t]*){5,}', ' ', text)
    # lone page numbers on their own line
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)
    return text


def is_garbage_text(text: str) -> bool:
    """Returns True if the extracted text is unusable and we should try something else."""
    if not text or len(text.strip()) < 50:
        return True

    # control characters = almost certainly a binary/image PDF
    ctrl = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if ctrl / len(text) > 0.1:
        return True

    # some PDFs extract as mostly whitespace with barely any real tokens
    # (e.g. scanned docs where pymupdf just returns blank lines)
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

    # proper unicode Nepali, definitely not garbage
    if devanagari / total > 0.15:
        return False

    # high suspect ratio = Preeti encoding
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
        log.warning(f"  pymupdf failed on {pdf_path.name}: {e}")
        return ""


def extract_text_preeti(raw_text: str) -> str:
    if _preeti_to_unicode is None:
        return ""
    try:
        return _preeti_to_unicode(raw_text)
    except Exception as e:
        log.warning(f"  Preeti conversion failed: {e}")
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
                # bail out if a single file is taking forever
                if time.time() - ocr_start > OCR_TIMEOUT:
                    log.warning(f"  OCR timeout on {pdf_path.name} at page {page_num+1}")
                    break
                try:
                    page = doc[page_num]
                    pix  = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))

                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4:
                        img = img[:, :, :3]  # drop alpha channel

                    results = _easyocr_reader.readtext(img, detail=0, paragraph=True)
                    text    = "\n".join(results)
                    if text.strip():
                        pages.append(text)

                except MemoryError:
                    log.warning(f"  OOM on page {page_num+1}, skipping")
                except Exception as e:
                    log.warning(f"  easyocr page {page_num+1} failed: {e}")

            doc.close()
            return "\n".join(pages)

        # tesseract fallback, only works if poppler is installed
        doc.close()
        if not POPPLER_PATH:
            log.warning(f"  no OCR available for {pdf_path.name} (no easyocr, no poppler)")
            return ""

        from pdf2image import convert_from_path
        import pytesseract

        log.info(f"  OCR via tesseract: {pdf_path.name}")
        for page_num in range(1, page_count + 1):
            try:
                images = convert_from_path(
                    pdf_path, dpi=150, poppler_path=POPPLER_PATH,
                    first_page=page_num, last_page=page_num,
                )
                text = pytesseract.image_to_string(images[0], lang="nep+eng")
                if text.strip():
                    pages.append(text)
            except MemoryError:
                log.warning(f"  OOM page {page_num}")
            except Exception as e:
                log.warning(f"  tesseract page {page_num}: {e}")

        return "\n".join(pages)

    except Exception as e:
        log.warning(f"  OCR totally gave up on {pdf_path.name}: {e}")
        return ""


def extract_text(pdf_path: Path, skip_ocr=False):
    """Try pymupdf first, then Preeti conversion, then OCR as last resort."""
    text = extract_text_pymupdf(pdf_path)

    if not is_garbage_text(text):
        return text, "pymupdf"

    if is_preeti(text):
        log.info(f"  Preeti detected: {pdf_path.name}")
        converted = extract_text_preeti(text)
        if converted.strip() and not is_garbage_text(converted):
            return converted, "preeti"
        log.warning(f"  Preeti conversion garbage, falling back to OCR: {pdf_path.name}")

    if skip_ocr:
        return "", "skipped"

    return extract_text_ocr(pdf_path), "ocr"


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    # 20% devanagari chars = Nepali. could probably be lower but 20% is safe
    return "np" if devanagari / max(len(text), 1) > 0.2 else "en"


def run_extraction(folders=None, skip_ocr=False):
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
                failed_files.append((pdf_path.name, "empty"))
                log.warning(f"  FAILED: {pdf_path.name}")
                continue

            text = clean_text(text)
            lang = detect_language(text)
            log.info(f"  method={method}  chars={len(text):,}  lang={lang}")

            folder_name = pdf_path.parent.name
            meta = {
                "source_file":       pdf_path.name,
                "folder":            folder_name,
                "category":          FOLDER_CATEGORIES.get(folder_name, "other"),
                "language":          lang,
                "char_count":        len(text),
                "extraction_method": method,
            }
            # utf-8-sig because some tools choke on raw utf-8 for Devanagari
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
            continue  # never stop the whole pipeline over one bad file

    log.info("=" * 60)
    log.info(f"extracted:{extracted}  skipped:{skipped}  failed:{failed}  crashed:{crashed}")
    if failed_files:
        for name, reason in failed_files:
            log.info(f"  problem: {name}, {reason}")
    log.info("=" * 60)

    return extracted + skipped


# ── chunking ──────────────────────────────────────────────────────────────────

SECTION_PATTERNS = [
    re.compile(r"^\s*(?:section|article|chapter|part)\s+\d+", re.I | re.M),
    re.compile(r"^\s*(?:दफा|धारा|अनुसूची|भाग)\s*\d*", re.M),
]


def find_section_breaks(text: str) -> list[int]:
    # not actually used in the main pipeline right now but keeping it
    # in case I want to do section-aware chunking later
    breaks = set()
    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            breaks.add(m.start())
    return sorted(breaks)


def chunk_text(text: str, meta: dict) -> list[dict]:
    words = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append({
            **meta,
            "chunk_index": len(chunks),
            "text":        " ".join(words[start:end]),
            "word_count":  end - start,
        })
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
        current = dict(chunks[i])
        merges  = 1

        while (
            merges < MAX_MERGE_COUNT
            and i + merges < len(chunks)
            and (current["word_count"] < MIN_CHUNK_WORDS
                 or chunks[i + merges]["word_count"] < MIN_CHUNK_WORDS)
        ):
            nxt = chunks[i + merges]
            current["text"]       = current["text"].rstrip() + "\n" + nxt["text"].lstrip()
            current["word_count"] += nxt["word_count"]
            merges += 1

        merged.append(current)
        i += merges

    # re-index after merging
    for idx, chunk in enumerate(merged):
        chunk["chunk_index"] = idx

    return merged


def run_chunking(folders=None):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = list(TEXT_DIR.glob("*.txt"))

    if folders:
        allowed = set()
        for folder in RAW_DIR.iterdir():
            if folder.is_dir() and folder.name in folders:
                allowed.update(p.stem for p in folder.glob("*.pdf"))
        txt_files = [f for f in txt_files if f.stem in allowed]

    log.info(f"chunking {len(txt_files)} texts...")
    total = 0
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
            chunks = merge_small_chunks(chunk_text(text, meta))

            if not chunks:
                continue

            with open(out_path, "w", encoding="utf-8-sig") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total += len(chunks)

        except Exception as e:
            failed += 1
            log.error(f"  chunking failed for {txt_path.name}: {e}")

    log.info(f"chunking done, {total} chunks  ({failed} failed)")
    return total


# ── embedding ─────────────────────────────────────────────────────────────────

def make_chunk_id(source_file: str, chunk_index: int) -> str:
    # md5 is fine here, we just need stable dedup IDs, not crypto
    return hashlib.md5(f"{source_file}_{chunk_index}".encode()).hexdigest()


def get_chunk_files(folders=None):
    chunk_files = list(CHUNKS_DIR.glob("*.jsonl"))
    if not folders:
        return chunk_files
    allowed = set()
    for folder in RAW_DIR.iterdir():
        if folder.is_dir() and folder.name in folders:
            allowed.update(p.stem for p in folder.glob("*.pdf"))
    return [f for f in chunk_files if f.stem in allowed]


def load_pending_chunks(chunk_files, existing_ids):
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
            log.error(f"  couldn't read {chunk_file.name}: {e}")
    return pending


def embed_and_ingest(pending, collection, model, batch_size):
    # e5 models need the "passage: " prefix for asymmetric retrieval
    # queries get "query: ", this is important, don't remove it
    texts = [f"passage: {c['text']}" for c in pending]

    log.info(f"encoding {len(texts)} chunks on {DEVICE.upper()}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # chromadb gets unhappy with batches > ~1000, 512 is safe
    chroma_batch = 512
    log.info("ingesting into chromadb...")
    for i in tqdm(range(0, len(pending), chroma_batch), desc="ingesting"):
        try:
            batch  = pending[i:i + chroma_batch]
            embeds = embeddings[i:i + chroma_batch].tolist()
            collection.add(
                ids        = [c["_id"] for c in batch],
                embeddings = embeds,
                documents  = [c["text"] for c in batch],
                metadatas  = [{
                    "source_file":       c["source_file"],
                    "folder":            c["folder"],
                    "category":          c["category"],
                    "language":          c["language"],
                    "chunk_index":       c["chunk_index"],
                    "word_count":        c["word_count"],
                    "extraction_method": c.get("extraction_method", "unknown"),
                } for c in batch],
            )
        except Exception as e:
            log.error(f"  chromadb batch {i} failed: {e}")
            continue


def run_embedding(batch_size=64, folders=None):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"loading {EMBED_MODEL} on {DEVICE.upper()}")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    if DEVICE == "cuda":
        log.info(f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    existing_ids = set(collection.get(include=[])["ids"])
    log.info(f"chromadb already has {len(existing_ids)} chunks")

    pending = load_pending_chunks(get_chunk_files(folders), existing_ids)
    log.info(f"{len(pending)} new chunks to embed")

    if not pending:
        log.info("nothing to do")
        return

    embed_and_ingest(pending, collection, model, batch_size)

    if DEVICE == "cuda":
        log.info(f"peak VRAM: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

    log.info(f"done, collection now has {collection.count()} chunks total")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CivicLens ingestion pipeline")
    parser.add_argument("--folder",       help="process only this folder (e.g. 11_laws_np)")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--chunk-only",   action="store_true")
    parser.add_argument("--embed-only",   action="store_true")
    parser.add_argument("--skip-ocr",     action="store_true", help="skip OCR fallback, much faster")
    parser.add_argument("--batch-size",   type=int, default=64, help="128 for GTX 1650, 256+ for newer cards")
    args = parser.parse_args()

    folders = {args.folder} if args.folder else None

    log.info(f"pipeline started, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    log.info("all done")


if __name__ == "__main__":
    main()