# CivicLens Nepal

A RAG-powered assistant for Nepal's laws, constitution, and governance documents. Ask questions in Nepali or English and get factual, cited answers pointing back to the exact source document.

**Live demo: coming soon**

![CivicLens Screenshot](https://github.com/user-attachments/assets/9006de28-c077-4231-bad1-eb7ca61b8eb7)

---

## Why this exists

Nepal's legal and governance documents are scattered across government websites, hard to search, and often only available as scanned PDFs in legacy Preeti font encoding. CivicLens indexes all of it and makes it queryable in plain language.

Ask things like:
- *भ्रष्टाचारको सजाय के हो?*
- *What does the constitution say about fundamental rights?*
- *मजदुरको न्यूनतम पारिश्रमिक कति हो?*

Every answer cites the exact document it came from, with a relevance score.

---

## What's indexed

> **Note:** This is an early experimental version. Data is not included in this repo — the raw PDFs are sourced from various Nepali government websites and are too large for GitHub. A download link will be added here once they're hosted somewhere reasonable (hopefully soon).

| Category | Documents |
|----------|-----------|
| Constitution | Nepal's constitution (English + Nepali) |
| Economic Surveys | 2002 – 2024 |
| Budget Speeches | 2020 – 2026 |
| PSC Syllabuses | 40+ civil service exam syllabuses |
| Civil Service Acts | Civil Service Act, Good Governance Act, Local Administration Act |
| Key Laws | 500+ Nepali laws (English and Nepali) |
| NPC Documents | Planning documents, SDG reports, annual reports |

Currently **67,079 chunks** indexed. Expected to grow as more documents are added.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `intfloat/multilingual-e5-base` |
| Vector store | ChromaDB |
| LLM | Llama 3.1 8B via Groq |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

Multilingual-e5-base was chosen specifically for its strong performance on both Devanagari and Latin script — most English-only embedding models struggle badly with Nepali text.

---

## Running locally

You need Python 3.10+ and a free [Groq API key](https://console.groq.com) (no credit card required).

```bash
git clone https://github.com/bibidhSubedi0/CivicLensNepal
cd CivicLensNepal

python -m venv cln_env
cln_env\Scripts\activate      # Windows
source cln_env/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run the ingestion pipeline to build the index (requires PDFs in `data/raw/`):
```bash
python pipeline.py
```

Start the server:
```bash
uvicorn server:app --port 8000
```

Open `http://localhost:8000`.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for all pipeline flags, folder structure, and advanced usage.

---

## License

MIT
