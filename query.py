import re
import sys
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from memory import ConversationMemory, expand_query

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.1-8b-instant"
CHROMA_DIR      = Path("data/chromadb")
COLLECTION_NAME = "civiclens_nepal"
EMBED_MODEL     = "intfloat/multilingual-e5-base"
TOP_K            = 6
PRIORITY_SOURCE  = "प्रतिवेदन_OK.pdf"  # set to None to disable
PRIORITY_BOOST   = 0.15

print("loading model...")
embedder   = SentenceTransformer(EMBED_MODEL)
chroma     = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_collection(COLLECTION_NAME)
client     = Groq(api_key=GROQ_API_KEY)
memory     = ConversationMemory(max_turns=6)
print(f"ready, {collection.count():,} chunks indexed\n")


def retrieve(query: str) -> list[dict]:
    # expand query with recent conversation context before embedding
    expanded = expand_query(query, memory)
    vec      = embedder.encode(f"query: {expanded}", normalize_embeddings=True).tolist()
    results  = collection.query(
        query_embeddings=[vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        source = meta.get("source_file", "unknown")
        score  = round(1 - dist, 4)
        if PRIORITY_SOURCE and source == PRIORITY_SOURCE:
            score = min(score + PRIORITY_BOOST, 1.0)
        chunks.append({
            "text":     doc,
            "source":   source,
            "category": meta.get("category", "unknown"),
            "language": meta.get("language", "unknown"),
            "score":    score,
        })

    if PRIORITY_SOURCE:
        chunks.sort(key=lambda c: c["score"], reverse=True)

    return chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[SOURCE {i}] {c['source']} (category: {c['category']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )

    # inject recent conversation history into the system context
    history = memory.get_context_string()
    history_block = f"\nCONVERSATION SO FAR:\n{history}\n" if history else ""

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
- Use the conversation history below (if any) to resolve follow-up questions and pronouns.
{history_block}
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

    # update memory with raw query (not expanded) and the answer
    memory.add("user", query)
    memory.add("assistant", answer)

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
        print("tip: type 'clear' to reset conversation memory\n")
        while True:
            try:
                q = input("Question: ").strip()
                if not q:
                    continue
                if q.lower() == "clear":
                    memory.clear()
                    print("memory cleared\n")
                    continue
                ask(q)
                print()
            except KeyboardInterrupt:
                print("\nbye!")
                break