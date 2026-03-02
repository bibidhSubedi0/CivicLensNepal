import re
import os
from pathlib import Path
from contextlib import asynccontextmanager

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from groq import Groq, APIStatusError
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.1-8b-instant"
CHROMA_DIR      = Path("data/chromadb")
COLLECTION_NAME = "civiclens_nepal"
EMBED_MODEL     = "intfloat/multilingual-e5-base"
TOP_K           = 6

WEBAPP = Path("WebApp")


def _read(filename: str) -> str:
    return (WEBAPP / filename).read_text(encoding="utf-8")


# ── startup ───────────────────────────────────────────────────────────────────

embedder    = None
collection  = None
groq_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, collection, groq_client
    print("loading embedding model...")
    # force cuda, on CPU this takes forever and the server feels broken
    embedder   = SentenceTransformer(EMBED_MODEL, device="cuda")
    chroma     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_collection(COLLECTION_NAME)
    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"ready, {collection.count():,} chunks indexed")
    yield


app = FastAPI(title="CivicLens Nepal", lifespan=lifespan)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_read("index.html"))

@app.get("/broke", response_class=HTMLResponse)
async def show_broke():
    return HTMLResponse(_read("broke.html"), status_code=503)

@app.get("/error", response_class=HTMLResponse)
async def show_error():
    return HTMLResponse(_read("error.html"), status_code=500)

@app.get("/favicon.ico")
async def favicon():
    # browsers spam this endpoint constantly, just return empty
    return Response(status_code=204)

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return HTMLResponse(_read("error.html"), status_code=404)

@app.exception_handler(500)
async def server_error(request: Request, exc):
    return HTMLResponse(_read("error.html"), status_code=500)


# ── retrieval + generation ────────────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    vec     = embedder.encode(f"query: {query}", normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source_file", "unknown"),
            "category": meta.get("category", "unknown"),
            "language": meta.get("language", "unknown"),
            "score":    round(1 - dist, 4),
        })
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[SOURCE {i}] {c['source']} (category: {c['category']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"""You are CivicLens, a helpful assistant for Nepali governance, law, and public policy.
            Answer the user's question using ONLY the sources provided below.

            Rules:
            - Never answer questions about current officeholders, recent events, or anything time-sensitive. Say you don't have that information.
            - Cite sources inline using [SOURCE N] whenever you use information from them.
            - If multiple sources support a point, cite all of them e.g. [SOURCE 1][SOURCE 3].
            - If the sources don't contain enough information, say so clearly.
            - If the question is in English, answer in English. If the question is in Nepali, answer in both:
            [ENG] English answer here
            [NEP] Nepali answer here
            - Be concise and factual.

            SOURCES:
            {context}

            QUESTION: {query}

            ANSWER:"""


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query(req: QueryRequest):
    try:
        chunks   = retrieve(req.question)
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": build_prompt(req.question, chunks)}],
        )
        answer = response.choices[0].message.content
        cited  = sorted(set(int(n) for n in re.findall(r'\[SOURCE (\d+)\]', answer)))

        sources = [
            {
                "index":    i,
                "source":   chunks[i-1]["source"],
                "category": chunks[i-1]["category"],
                "language": chunks[i-1]["language"],
                "score":    chunks[i-1]["score"],
            }
            for i in cited if i <= len(chunks)
        ]

        return {"answer": answer, "sources": sources}

    except APIStatusError:
        # groq free tier quota, redirect to the broke page
        return JSONResponse({"redirect": "/broke"}, status_code=402)
    except Exception:
        return JSONResponse({"redirect": "/error"}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": collection.count(), "model": GROQ_MODEL}


@app.get("/stats")
async def stats():
    return {
        "collection":   COLLECTION_NAME,
        "total_chunks": collection.count(),
        "embed_model":  EMBED_MODEL,
        "llm":          GROQ_MODEL,
        "top_k":        TOP_K,
    }