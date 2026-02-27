# CivicLens Nepal 

A RAG-powered assistant that lets you ask questions about Nepal's laws, constitution, and governance documents — in Nepali or English — and get factual answers with citations pointing to the exact source documents.

**Live: NOT_RIGHT_NOW_UNDER_CONSTRUCTION**

<img width="1894" height="940" alt="SS" src="https://github.com/user-attachments/assets/9006de28-c077-4231-bad1-eb7ca61b8eb7" />

---

## Why

Nepal's legal and governance documents are scattered, hard to search, and often only available as scanned PDFs in legacy fonts. CivicLens indexes all of it and makes it queryable in plain language.

Ask things like:
- *भ्रष्टाचारको सजाय के हो?*
- *What does the constitution say about fundamental rights?*
- *मजदुरको न्यूनतम पारिश्रमिक कति हो?*

Every answer cites the exact document it came from.

---

## Documents indexed

~FOR THIS VERY EXPREMENTAL VERSION : 9,500 chunks (Expected to be 100,000+) across Nepal's constitution, budget speeches, economic surveys, PSC syllabuses, civil service acts, key legislation (Nepali and English), and NPC planning documents.

---

## Stack

- **Embeddings** — `intfloat/multilingual-e5-base` (supports Nepali + English)
- **Vector store** — ChromaDB
- **LLM** — Llama 3.1 8B via Groq
- **Backend** — FastAPI
- **Frontend** — vanilla HTML/CSS/JS, no frameworks

---

## Running locally

You need Python 3.10+, and a free [Groq API key](https://console.groq.com) (no credit card).

```bash
git clone https://github.com/bibidhSubedi0/CivicLensNepal
cd civiclens-nepal
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

Add your key to a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run the ingestion pipeline once to build the index:
```bash
python ingest.py
```

Then start the server:
```bash
uvicorn server:app --port 8000
```

Open `http://localhost:8000`.

---

## License

MIT
