import numpy as np
import faiss
from typing import List
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
import requests
import os

# === Set API Key ===
EURI_API_KEY= "euri-....."

# === Sample Chunks ===
sample_chunks = [
    "LangChain is a framework to build applications using LLMs.",
    "FAISS is used for storing and searching embeddings quickly.",
    "Open-source GPT models power many RAG systems.",
    "Embedding quality affects semantic search accuracy.",
    "Recursive chunking provides contextual continuity."
]

# === Task 17: Get EURI embeddings via API ===
def get_euri_embeddings(texts: List[str]) -> np.ndarray:
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": texts,
        "model": "text-embedding-3-small"
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    vectors = [np.array(item["embedding"]) for item in data["data"]]
    return np.array(vectors)

# === HuggingFace Embedding ===
def get_hf_embeddings(texts: List[str]) -> np.ndarray:
    hf_model = SentenceTransformer('all-mpnet-base-v2')
    return hf_model.encode(texts, convert_to_numpy=True)

# === Task 18: Manual Cosine Similarity ===
def manual_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Task 19: FAISS Search ===
def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def search_faiss(index, query_vec, k=3):
    D, I = index.search(np.array([query_vec]), k)
    print(f"\n🔍 Top {k} Most Similar Chunks:")
    for idx, dist in zip(I[0], D[0]):
        print(f"Chunk #{idx+1}: '{sample_chunks[idx]}' | Distance: {dist:.4f}")

# === Task 20: Compare EURI vs HF with Heatmap ===
def compare_models(texts: List[str]):
    print("\n🔹 Getting EURI Embeddings...")
    euri_vecs = get_euri_embeddings(texts)

    print("🔹 Getting HuggingFace Embeddings...")
    hf_vecs = get_hf_embeddings(texts)

    # Cosine similarities for first 3 chunks
    euri_sim = sklearn_cosine(euri_vecs[:3])
    hf_sim = sklearn_cosine(hf_vecs[:3])

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    models = ["EURI", "HuggingFace"]
    sims = [euri_sim, hf_sim]

    for ax, sim, title in zip(axes, sims, models):
        sns.heatmap(sim, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(f"{title} Cosine Similarity")
    plt.tight_layout()
    os.makedirs("test_data", exist_ok=True)
    plt.savefig("test_data/embedding_similarity_comparison.png")
    plt.show()
    print("✅ Heatmap saved as: test_data/embedding_similarity_comparison.png")

# === Run All ===
if __name__ == "__main__":
    print("\n🔹 Task 17: Generate EURI Embeddings")
    euri_vecs = get_euri_embeddings(sample_chunks)
    for i, v in enumerate(euri_vecs):
        print(f"Embedding {i+1} shape: {v.shape}")

    print("\n🔹 Task 18: Manual vs sklearn Cosine")
    sim_manual = manual_cosine(euri_vecs[0], euri_vecs[1])
    sim_sklearn = sklearn_cosine([euri_vecs[0]], [euri_vecs[1]])[0][0]
    print(f"Manual Cosine: {sim_manual:.4f} | sklearn Cosine: {sim_sklearn:.4f}")

    print("\n🔹 Task 19: FAISS Search")
    index = build_faiss_index(euri_vecs)
    search_faiss(index, euri_vecs[0])

    print("\n🔹 Task 20: Compare EURI vs HuggingFace Embeddings")
    compare_models(sample_chunks)
