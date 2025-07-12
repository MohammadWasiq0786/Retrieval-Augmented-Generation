from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# ========== Token counter using tiktoken ==========
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# ========== Step 1: Load PDF and split ==========
def load_pdf_and_split(file_path: str, chunk_size=500, overlap=100) -> List:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_docs = text_splitter.split_documents(docs)

    # Inject metadata
    for idx, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = idx + 1
        doc.metadata["document_type"] = "pdf"
        doc.metadata["source"] = file_path

    print(f"[+] Injected metadata into {len(split_docs)} chunks.")
    for i in range(min(3, len(split_docs))):
        print(f"\nChunk {i+1} Metadata: {split_docs[i].metadata}")
    return split_docs

# ========== Step 2: Filter chunks ==========
def filter_chunks(docs: List, min_words=30, max_tokens=400) -> List:
    filtered = []
    for doc in docs:
        word_count = len(doc.page_content.split())
        token_count = count_tokens(doc.page_content)
        if word_count >= min_words and token_count <= max_tokens:
            filtered.append(doc)
    print(f"[Filter] From {len(docs)} → {len(filtered)} chunks after filtering.")
    return filtered

# ========== Sample Usage ==========
if __name__ == "__main__":
    pdf_path = "test_data/large_sample.pdf"  # Ensure you have a sample PDF here
    chunks = load_pdf_and_split(pdf_path)
    filtered_chunks = filter_chunks(chunks)
