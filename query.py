import re
import sys
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.1-8b-instant"
CHROMA_DIR      = Path("data/chromadb")
COLLECTION_NAME = "civiclens_nepal"
EMBED_MODEL     = "intfloat/multilingual-e5-base"
TOP_K           = 6

print("loading model...")
embedder   = SentenceTransformer(EMBED_MODEL)
chroma     = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_collection(COLLECTION_NAME)
client     = Groq(api_key=GROQ_API_KEY)
print(f"ready, {collection.count():,} chunks indexed\n")


def retrieve(query: str) -> list[dict]:
    # e5 asymmetric retrieval: queries get "query: " prefix, passages get "passage: "
    # skipping this prefix tanks retrieval quality noticeably
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


def ask(query: str) -> None:
    chunks   = retrieve(query)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": build_prompt(query, chunks)}],
    )
    answer = response.choices[0].message.content

    print("\n" + "=" * 60)
    print(answer)

    cited = sorted(set(int(n) for n in re.findall(r'\[SOURCE (\d+)\]', answer)))
    if cited:
        print("\n--- sources ---")
        for i in cited:
            if i <= len(chunks):
                c = chunks[i - 1]
                print(f"  [{i}] {c['source']}  ({c['category']}, {c['language']}, score={c['score']})")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("CivicLens Nepal: type your question, Ctrl+C to exit\n")
        while True:
            try:
                q = input("Question: ").strip()
                if q:
                    ask(q)
                print()
            except KeyboardInterrupt:
                print("\nbye!")
                break