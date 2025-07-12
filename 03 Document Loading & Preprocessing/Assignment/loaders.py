from langchain.document_loaders import (TextLoader, 
                                        CSVLoader, 
                                        UnstructuredWordDocumentLoader, 
                                        NotionDirectoryLoader, 
                                        WebBaseLoader)
from pathlib import Path
import json

def load_txt_file(file_path: str):
    loader = TextLoader(file_path)
    docs = loader.load()
    print("First 500 characters:\n", docs[0].page_content[:500])
    print("Metadata:\n", docs[0].metadata)

def load_csv_file(file_path: str, column_name: str = "description"):
    loader = CSVLoader(file_path=file_path, source_column=column_name)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

def load_docx_with_metadata(file_path: str):
    loader = UnstructuredWordDocumentLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({
            "source": "Assignment",
            "date": "2025-07-11",
            "page": 1
        })
    print("Sample with injected metadata:", docs[0].metadata)

def load_notion_export(notion_path: str):
    loader = NotionDirectoryLoader(notion_path)
    docs = loader.load()
    final_docs = [doc for doc in docs if 'final' in doc.metadata.get("tags", [])]
    print("Titles of final-tagged pages:")
    for doc in final_docs:
        print(doc.metadata.get("title", "Untitled"))

def load_website_and_save(url: str, output_path: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = docs[0].page_content
    cleaned_text = '\n'.join([line for line in text.splitlines() if "script" not in line.lower() and "advertisement" not in line.lower()])
    Path(output_path).write_text(cleaned_text)
    print(f"Saved cleaned content to {output_path}")

if __name__ == "__main__":
    load_txt_file("test_data/large_sample.txt")
    load_csv_file("test_data/large_sample.csv")
    load_docx_with_metadata("test_data/large_sample.docx")
    load_notion_export("test_data/notion_export")
    load_website_and_save("https://en.wikipedia.org/wiki/Generative_artificial_intelligence", "test_data/web_output.txt")
