import os
import subprocess

print("📦 Starting LangChain Assignment Project...\n")

print("📁 Generating large test data...")
subprocess.run(["python3", "generate_test_data.py"])

print("\n📂 Testing all loaders...")
subprocess.run(["python3", "loaders.py"])

print("\n🔪 Running chunking strategies...")
subprocess.run(["python3", "chunking.py"])

print("\n🧾 Injecting metadata and filtering chunks...")
subprocess.run(["python3", "metadata_filter.py"])

print("\n✂️ Splitting text using 3 strategies...")
subprocess.run(["python3", "text_splitters.py"])

print("\n📊 Chunk size analysis and visualization...")
subprocess.run(["python3", "chunk_analysis.py"])

print("\n📡 Generating embeddings and computing similarity...")
subprocess.run(["python3", "embeddings.py"])

print("\n✅ LangChain assignment pipeline completed successfully!")