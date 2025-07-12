import time
import nltk
from pathlib import Path
from typing import List, Tuple
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from nltk.tokenize import sent_tokenize
from langchain.schema import Document

nltk.download('punkt')

def read_large_text(path: str) -> str:
    return Path(path).read_text()

def recursive_chunking(text: str, chunk_size=500, overlap=100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    print(f"[Recursive] Generated {len(chunks)} chunks.")
    print("\nSample with Overlap:")
    for i in range(5):
        print(f"\nChunk {i+1}:\n", chunks[i][:300])
    return chunks

def token_chunking(text: str, chunk_size=256, overlap=50) -> List[str]:
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    print(f"[Token] Total Chunks: {len(chunks)}")
    return chunks

def sentence_chunking(text: str, sentences_per_chunk=3) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    print(f"[Semantic] Created {len(chunks)} semantic chunks.")
    avg_len = sum(len(chunk.split()) for chunk in chunks) // len(chunks)
    print(f"Average semantic chunk length: {avg_len} words")
    return chunks

def auto_chunk_by_type(path: str) -> List[str]:
    ext = Path(path).suffix
    text = read_large_text(path)
    if ext == ".txt":
        return recursive_chunking(text)
    elif ext == ".csv":
        return token_chunking(text)
    elif ext == ".docx":
        return sentence_chunking(text)
    else:
        print("Unknown type, using default recursive splitter.")
        return recursive_chunking(text)

def compare_strategies(text: str) -> None:
    strategies = {
        "Recursive": lambda: recursive_chunking(text),
        "Token": lambda: token_chunking(text),
        "Semantic": lambda: sentence_chunking(text, 3),
    }
    results = {}
    for name, func in strategies.items():
        start = time.time()
        chunks = func()
        end = time.time()
        total_tokens = sum(len(c.split()) for c in chunks)
        results[name] = {
            "chunks": len(chunks),
            "avg_token_len": total_tokens // len(chunks),
            "time_sec": round(end - start, 3)
        }
    print("\n📊 Strategy Comparison:")
    for name, metrics in results.items():
        print(f"{name} → Chunks: {metrics['chunks']}, Avg Tokens: {metrics['avg_token_len']}, Time: {metrics['time_sec']}s")

# Sample usage
if __name__ == "__main__":
    path = "test_data/web_output.txt"
    full_text = read_large_text(path)

    print("\n🌀 Recursive Chunking Test")
    recursive_chunking(full_text)

    print("\n📦 Token-Based Chunking Test")
    token_chunking(full_text)

    print("\n🧠 Sentence-Based Chunking Test")
    sentence_chunking(full_text)

    print("\n⚡ Auto Chunking by File Type")
    auto_chunk_by_type(path)

    print("\n📈 Chunking Strategy Comparison")
    compare_strategies(full_text)
