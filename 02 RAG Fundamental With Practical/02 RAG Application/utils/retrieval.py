import faiss
import numpy as np
import pickle
import os

from utils.embedding import get_embedding
from utils.chunking import chunk_text

def load_faiss_index():
    index_path = "faiss_store/index.faiss"
    mapping_path = "faiss_store/chunk_mapping.pkl"

    valid = os.path.exists(index_path) and os.path.getsize(index_path) > 0
    valid = valid and os.path.exists(mapping_path) and os.path.getsize(mapping_path) > 0

    if valid:
        try:
            index = faiss.read_index(index_path)
            with open(mapping_path, "rb") as f:
                chunk_mapping = pickle.load(f)
            return index, chunk_mapping
        except Exception as e:
            print("Corrupted index detected. Rebuilding...", e)

    print("Generating new FAISS index from founder_story.txt...")

    with open("data/founder_story.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    chunk_mapping = []
    all_embeddings = []

    for chunk in chunks:
        emb = get_embedding(chunk)
        all_embeddings.append(emb)
        chunk_mapping.append(chunk)

    all_embeddings = np.array(all_embeddings).astype("float32")
    dimension = all_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)

    print("Index built and saved.")
    return index, chunk_mapping


def retrieve_chunks(query, index, chunk_mapping, k=3):
    query_vec = get_embedding(query)
    distances, indices = index.search(np.array([query_vec]).astype("float32"), k)
    return [chunk_mapping[i] for i in indices[0]]
