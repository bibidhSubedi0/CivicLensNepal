import re
import time
import argparse
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

SITES = {
    "npc": {
        "folder": "09_npc_documents",
        "delay": 2,
        "index_urls": [
            "https://npc.gov.np/category/626/",
            "https://npc.gov.np/category/634/",
            "https://npc.gov.np/category/639/",
            "https://npc.gov.np/category/640/",
            "https://npc.gov.np/category/642/",
            "https://npc.gov.np/category/643/",
            "https://npc.gov.np/category/projects-of-national-pride/",
        ],
        "list_link_pattern": r"/content/\d+/",
        "pdf_selector": "a.df-ui-download",
        "lang": "np",
        "max_pages": 20,
    },
    "mof": {
        "folder": "02_economic_survey",
        "delay": 2,
        "index_urls": [
            "https://mof.gov.np/en/archive-documents-detail-page/publication-1/economic-survey",
            "https://mof.gov.np/en/archive-documents-detail-page/publication-1/budget-speech",
        ],
        "list_link_pattern": r"/en/archive-documents-detail-page/",
        "pdf_selector": "a.df-ui-download",
        "lang": "en",
    },
    "lawcomm_en": {
        "folder": "10_laws_en",
        "delay": 1.5,
        "index_urls": [
            "https://www.lawcommission.gov.np/en/acts",
            "https://www.lawcommission.gov.np/en/regulations",
            "https://www.lawcommission.gov.np/en/rules",
        ],
        "list_link_pattern": r"/en/(acts|regulations|rules)/\d+",
        "pdf_selector": "a[href$='.pdf'], a.df-ui-download",
        "lang": "en",
        "max_pages": 200,
    },
    "lawcomm_np": {
        "folder": "11_laws_np",
        "delay": 1.5,
        "index_urls": [
            "https://www.lawcommission.gov.np/np/acts",
            "https://www.lawcommission.gov.np/np/regulations",
        ],
        "list_link_pattern": r"/np/(acts|regulations|rules)/\d+",
        "pdf_selector": "a[href$='.pdf'], a.df-ui-download",
        "lang": "np",
        "max_pages": 200,
    },
    "lawrepo_np": {
        "folder": "11_laws_np",
        "delay": 1.5,
        "direct": True,
        "page_style": "slash",
        "verify_ssl": False,
        "index_urls": [
            "https://repository.lawcommission.gov.np/np/documents/",
            "https://repository.lawcommission.gov.np/np/document-category/संविधान/",
            "https://repository.lawcommission.gov.np/np/document-category/मौजुदा-कानूनहरु-ऐनहरु/",
            "https://repository.lawcommission.gov.np/np/document-category/मौजुदा-कानूनहरु-अध्यादेश/",
            "https://repository.lawcommission.gov.np/np/document-category/मौजुदा-कानूनहरु-गठन-आदेश/",
            "https://repository.lawcommission.gov.np/np/document-category/मौजुदा-कानूनहरु-नियमहरु/",
            "https://repository.lawcommission.gov.np/np/document-category/मौजुदा-कानूनहरु-नीतिहरु/",
            "https://repository.lawcommission.gov.np/np/document-category/निर्देशिका/",
            "https://repository.lawcommission.gov.np/np/document-category/मौजुदा-कानूनहरु-अन्तराष्/",
            "https://repository.lawcommission.gov.np/np/document-category/पुराना-कानून-संग्रह-नियम/",
            "https://repository.lawcommission.gov.np/np/document-category/पुराना-कानून-संग्रह-ऐतिह/",
            "https://repository.lawcommission.gov.np/np/document-category/आयोगबाट-तर्जुमा-गरिएको/",
            "https://repository.lawcommission.gov.np/np/document-category/पुराना-कानून-संग्रह-संवि/",
        ],
        "list_link_pattern": r"",
        "pdf_selector": "a[href$='.pdf']",
        "lang": "np",
        "max_pages": 200,
    },
    "psc": {
        "folder": "04_psc_syllabuses",
        "delay": 2,
        "index_urls": ["https://psc.gov.np/en/syllabus"],
        "list_link_pattern": r"/en/syllabus/\d+",
        "pdf_selector": "a.df-ui-download, a[href*='/uploads/course/']",
        "lang": "np",
    },
}


def fetch(url, session, verify_ssl=True):
    try:
        r = session.get(url, timeout=30, verify=verify_ssl)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        log.warning(f"GET failed: {url} — {e}")
        return None


def clean_filename(url, prefix=""):
    name = unquote(urlparse(url).path.split("/")[-1])
    name = re.sub(r"_[a-z0-9]{6,8}\.pdf$", ".pdf", name, flags=re.I)
    name = re.sub(r"\s+", "-", name)
    if prefix and not name.lower().startswith(prefix.lower()):
        name = f"{prefix}-{name}"
    return name


def save_pdf(url, dest, session, verify_ssl=True):
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"  SKIP  {dest.name}")
        return True
    if dest.exists():
        dest.unlink()
    try:
        r = session.get(url, timeout=120, stream=True, verify=verify_ssl)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "pdf" not in ct and not url.lower().endswith(".pdf"):
            log.warning(f"  SKIP  {dest.name} — unexpected content type: {ct}")
            return False
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        tmp.rename(dest)
        log.info(f"  OK    {dest.name}  ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        log.warning(f"  FAIL  {dest.name} — {e}")
        if dest.exists():
            dest.unlink()
        return False


def get_next_page(soup, current, index_url, page_num, style):
    for a in soup.find_all("a", href=True):
        txt = a.get_text(strip=True)
        if "next" in a.get("rel", []) or txt in ("Next", "›", "»", ">") or "next" in a.get("aria-label", "").lower():
            return urljoin(current, a["href"])
    if style == "slash":
        return index_url.rstrip("/") + f"/page/{page_num}/"
    sep = "&" if "?" in index_url else "?"
    return f"{index_url}{sep}page={page_num}"


def find_pdfs(soup, doc_url, sel):
    found = []
    for tag in soup.select(sel):
        href = tag.get("href", "").strip()
        if href:
            found.append(urljoin(doc_url, href))
    if found:
        return found
    for tag in soup.find_all("a", href=re.compile(r"giwmscdn(one|two|three)\.gov\.np.*\.pdf", re.I)):
        found.append(urljoin(doc_url, tag["href"].strip()))
    if found:
        return found
    for tag in soup.find_all(["iframe", "embed"], src=re.compile(r"\.pdf", re.I)):
        src = tag.get("src", "").strip()
        if src:
            found.append(urljoin(doc_url, src))
    if found:
        return found
    for script in soup.find_all("script"):
        for m in re.findall(r"https?://[^\s\"']+\.pdf", script.string or "", re.I):
            found.append(m.strip())
    if found:
        return found
    for tag in soup.find_all(True):
        for attr in ("data-src", "data-url", "data-file", "data-href"):
            val = tag.get(attr, "").strip()
            if val.lower().endswith(".pdf"):
                found.append(urljoin(doc_url, val))
    if found:
        return found
    for tag in soup.find_all("a", href=re.compile(r"\.pdf(\?.*)?$", re.I)):
        found.append(urljoin(doc_url, tag["href"].strip()))
    return found


def scrape_site(key, cfg, base_dir, dry_run=False):
    dest_dir = base_dir / cfg["folder"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    delay = cfg.get("delay", 2)
    max_pages = cfg.get("max_pages", 10)
    verify_ssl = cfg.get("verify_ssl", True)
    style = cfg.get("page_style", "query")
    direct = cfg.get("direct", False)

    log.info(f"\n{'='*55}")
    log.info(f"{key}  →  {cfg['folder']}" + ("  [direct]" if direct else ""))
    log.info("=" * 55)

    ok = fail = skip = 0

    if direct:
        for index_url in cfg["index_urls"]:
            page_url = index_url
            page_num = 1
            while page_url and page_num <= max_pages:
                soup = fetch(page_url, session, verify_ssl)
                if not soup:
                    break
                tags = soup.select(cfg["pdf_selector"])
                count = 0
                for tag in tags:
                    pdf_url = urljoin(page_url, tag["href"].strip())
                    fname = clean_filename(pdf_url)
                    dest = dest_dir / fname
                    if dry_run:
                        log.info(f"  DRY   {fname}\n        {pdf_url}")
                        skip += 1
                    else:
                        if save_pdf(pdf_url, dest, session, verify_ssl):
                            ok += 1
                        else:
                            fail += 1
                        time.sleep(delay)
                    count += 1
                log.info(f"  p{page_num}: {count} PDFs  [{page_url}]")
                if count == 0:
                    break
                page_num += 1
                page_url = get_next_page(soup, page_url, index_url, page_num, style)
                time.sleep(delay)
    else:
        seen_docs: set[str] = set()
        pattern = re.compile(cfg["list_link_pattern"])

        for index_url in cfg["index_urls"]:
            page_url = index_url
            page_num = 1
            while page_url and page_num <= max_pages:
                soup = fetch(page_url, session, verify_ssl)
                if not soup:
                    break
                new = 0
                for a in soup.find_all("a", href=pattern):
                    full = urljoin(page_url, a["href"].strip())
                    if full not in seen_docs:
                        seen_docs.add(full)
                        new += 1
                log.info(f"  p{page_num}: {new} new links (total {len(seen_docs)})  [{page_url}]")
                if new == 0:
                    break
                page_num += 1
                page_url = get_next_page(soup, page_url, index_url, page_num, style)
                time.sleep(delay)

        log.info(f"\n{len(seen_docs)} doc pages found, fetching PDFs...")

        for doc_url in sorted(seen_docs):
            time.sleep(delay)
            soup = fetch(doc_url, session, verify_ssl)
            if not soup:
                fail += 1
                continue

            pdfs = list(dict.fromkeys(find_pdfs(soup, doc_url, cfg["pdf_selector"])))

            if not pdfs:
                log.warning(f"  no PDF on: {doc_url}")
                fail += 1
                continue

            h1 = soup.find("h1")
            prefix = ""
            if h1:
                prefix = re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", h1.get_text()).strip())[:40]

            for pdf_url in pdfs:
                fname = clean_filename(pdf_url, prefix)
                dest = dest_dir / fname
                if dry_run:
                    log.info(f"  DRY   {fname}\n        {pdf_url}")
                    skip += 1
                else:
                    if save_pdf(pdf_url, dest, session, verify_ssl):
                        ok += 1
                    else:
                        fail += 1
                    time.sleep(delay)

    log.info(f"\n[{key}] done  ok:{ok}  skip:{skip}  fail:{fail}")
    return ok, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", help="run one site (e.g. npc, lawrepo_np)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    if args.site and args.site not in SITES:
        print(f"unknown site '{args.site}'. options: {', '.join(SITES)}")
        return

    base = Path(args.data_dir)
    base.mkdir(parents=True, exist_ok=True)
    targets = {args.site: SITES[args.site]} if args.site else SITES

    total_ok = total_fail = 0
    for key, cfg in targets.items():
        ok, fail = scrape_site(key, cfg, base, dry_run=args.dry_run)
        total_ok += ok
        total_fail += fail

    print(f"\ndone. ok:{total_ok}  fail:{total_fail}")


if __name__ == "__main__":
    main()