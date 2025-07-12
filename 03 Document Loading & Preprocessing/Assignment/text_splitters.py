import time
import nltk
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    NLTKTextSplitter,
)
from pathlib import Path
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# ========= Utility to Load Large Text File =========
def load_text(path: str) -> str:
    return Path(path).read_text()

# ========= Task 13: Compare Built-in Splitters =========
def compare_splitters(text: str):
    splitters = {
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100),
        "TokenTextSplitter": TokenTextSplitter(chunk_size=256, chunk_overlap=50),
        "NLTKTextSplitter": NLTKTextSplitter(chunk_size=400)
    }

    print("📊 Splitter Comparison:")
    for name, splitter in splitters.items():
        start = time.time()
        chunks = splitter.split_text(text)
        end = time.time()
        total_tokens = sum(len(chunk.split()) for chunk in chunks)
        avg_len = total_tokens // len(chunks)
        print(f"{name}: Chunks={len(chunks)}, Avg Words={avg_len}, Time={round(end-start, 3)}s")

# ========= Task 14: Custom NLTK Splitter with Overlap =========
def custom_sentence_splitter(text: str, sentences_per_chunk=4, overlap=1):
    sentences = sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + sentences_per_chunk]
        chunks.append(" ".join(chunk))
        i += (sentences_per_chunk - overlap)
    print(f"\n✅ Custom Splitter: {len(chunks)} chunks created (4-sentence chunks, 1 overlap).")
    print("\n📌 Sample Chunks:")
    for idx in range(min(3, len(chunks))):
        print(f"\n--- Chunk {idx+1} ---\n{chunks[idx]}")
    return chunks

# ========= Entry Point =========
if __name__ == "__main__":
    path = "test_data/web_output.txt"
    text = load_text(path)

    print("\n📍 Task 13: Compare Recursive, Token, and NLTK Splitters")
    compare_splitters(text)

    print("\n📍 Task 14: Custom NLTK Sentence Splitter with Overlap")
    custom_sentence_splitter(text)
