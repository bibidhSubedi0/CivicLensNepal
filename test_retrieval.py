import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")
client = chromadb.PersistentClient(path="data/chromadb")
col = client.get_collection("civiclens_nepal")

queries = [
    "what are the fundamental rights in the constitution?",
    "मौलिक अधिकार के हुन्?", 
    "what topics are in the section officer exam?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print("="*60)
    embedding = model.encode([f"query: {query}"], normalize_embeddings=True).tolist()
    results = col.query(query_embeddings=embedding, n_results=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n  result {i+1} — {meta['source_file']}  chunk:{meta['chunk_index']}  lang:{meta['language']}")
        print(f"  {doc[:250]}")