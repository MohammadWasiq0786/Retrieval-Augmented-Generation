# Vector Database Comparison

| Feature                | **FAISS**           | **ChromaDB**                   | **Pinecone**          | **Weaviate**                     | **Qdrant**                   |
| ---------------------- | ------------------- | ------------------------------ | --------------------- | -------------------------------- | ---------------------------- |
| **Type**               | Library             | Library + DB                   | Managed Cloud DB      | Managed DB + OSS                 | Managed DB + OSS             |
| **Hosting**            | Local               | Local (some cloud support)     | Cloud-only            | Cloud & Self-host                | Cloud & Self-host            |
| **Persistent Storage** | No (in-memory)      | Yes                            | Yes                   | Yes                              | Yes                          |
| **Scalability**        | Manual              | Limited                        | Auto-scaling          | Auto-scaling                     | Auto-scaling                 |
| **API Access**         | No REST API         | Python API                     | REST/gRPC API         | REST/gRPC API                    | REST/gRPC API                |
| **Indexing Options**   | IVF, HNSW, PQ, Flat | HNSW                           | Proprietary           | HNSW, Flat, IVF                  | HNSW, IVF, PQ                |
| **Metadata Filtering** | Manual              | Built-in                       | Built-in              | Built-in                         | Built-in                     |
| **Replication**        | No                  | No                             | Yes                   | Yes                              | Yes                          |
| **Embeddings Storage** | Vectors only        | Vectors + Metadata + Documents | Vectors + Metadata    | Vectors + Metadata + Schema      | Vectors + Payload            |
| **Integrations**       | Custom              | LangChain, LlamaIndex          | LangChain, LlamaIndex | LangChain, LlamaIndex            | LangChain, LlamaIndex        |
| **License**            | MIT                 | Apache 2.0                     | Proprietary SaaS      | Apache 2.0                       | Apache 2.0                   |
| **Best For**           | Fast local search   | Embeddings + metadata          | Production SaaS       | Knowledge graphs & hybrid search | Vector search with filtering |
