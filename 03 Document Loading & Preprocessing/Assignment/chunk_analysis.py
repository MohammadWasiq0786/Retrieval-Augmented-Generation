import matplotlib.pyplot as plt
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Tuple
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize


# ---------- Utility ----------
def load_text(path: str) -> str:
    return Path(path).read_text()

def token_chunking(text: str, token_size: int, overlap: int = 20) -> List[str]:
    splitter = TokenTextSplitter(chunk_size=token_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def word_chunking(text: str, word_size: int = 200) -> List[str]:
    words = word_tokenize(text)
    chunks = [' '.join(words[i:i+word_size]) for i in range(0, len(words), word_size)]
    return chunks

# ---------- Task 15 ----------
def analyze_token_chunks(text: str):
    print("📊 Chunk Counts for Various Token Sizes:")
    token_sizes = [128, 256, 512, 1024]
    for size in token_sizes:
        chunks = token_chunking(text, token_size=size)
        avg_len = sum(len(c.split()) for c in chunks) // len(chunks)
        print(f"Token size: {size} → Chunks: {len(chunks)}, Avg Word Length: {avg_len}")
        print(f"Sample Chunk [{size}]: {chunks[0][:150]}...\n")

# ---------- Task 16 ----------
def plot_comparison(text: str):
    token_sizes = [128, 256, 512, 1024]
    token_counts = []
    token_avgs = []

    word_sizes = [100, 200, 400, 800]
    word_counts = []
    word_avgs = []

    for size in token_sizes:
        chunks = token_chunking(text, token_size=size)
        token_counts.append(len(chunks))
        token_avgs.append(sum(len(c.split()) for c in chunks) // len(chunks))

    for size in word_sizes:
        chunks = word_chunking(text, word_size=size)
        word_counts.append(len(chunks))
        word_avgs.append(sum(len(c.split()) for c in chunks) // len(chunks))

    # Plot chunk count comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar([str(s) for s in token_sizes], token_counts, color="skyblue", label="Token-based")
    plt.bar([str(s) for s in word_sizes], word_counts, color="lightgreen", alpha=0.6, label="Word-based")
    plt.title("Number of Chunks")
    plt.xlabel("Chunk Size (Token/Word)")
    plt.ylabel("Chunk Count")
    plt.legend()

    # Plot average size comparison
    plt.subplot(1, 2, 2)
    plt.plot(token_sizes, token_avgs, marker='o', label="Token-based", color="blue")
    plt.plot(word_sizes, word_avgs, marker='x', label="Word-based", color="green")
    plt.title("Average Words per Chunk")
    plt.xlabel("Chunk Size")
    plt.ylabel("Avg Words")
    plt.legend()

    plt.tight_layout()
    plt.savefig("test_data/chunk_size_comparison.png")
    plt.show()
    print("✅ Chart saved as: test_data/chunk_size_comparison.png")

# ---------- Main ----------
if __name__ == "__main__":
    text = load_text("test_data/web_output.txt")

    print("\n📍 Task 15: Chunking with Token Sizes")
    analyze_token_chunks(text)

    print("\n📍 Task 16: Visual Comparison (Word vs Token)")
    plot_comparison(text)
