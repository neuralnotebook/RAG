# RAG Pipeline — Detailed Notes
> Source: YouTube transcript (mid-section) | Topic: Building a RAG pipeline from scratch

---

## Table of Contents
1. [Where We Left Off — Document Loaders Recap](#1-where-we-left-off)
2. [The Full RAG Pipeline Overview](#2-full-rag-pipeline-overview)
3. [Step 1 — Data Ingestion (PDF Loading)](#3-step-1--data-ingestion)
4. [Step 2 — Chunking (Text Splitting)](#4-step-2--chunking)
5. [Step 3 — Embeddings](#5-step-3--embeddings)
6. [Step 4 — Vector Store (ChromaDB)](#6-step-4--vector-store-chromadb)
7. [Step 5 — RAG Retriever](#7-step-5--rag-retriever)
8. [Step 6 — Augmented Generation (LLM Integration Preview)](#8-step-6--augmented-generation)
9. [Key Code Patterns](#9-key-code-patterns)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Where We Left Off

- Covered **LangChain Document Loaders** — PyPDF, PyMuPDF, TextLoader
- Every loader returns a **`Document` object** regardless of file type
- `Document` structure = `{ page_content: str, metadata: dict }`
- Metadata auto-includes: `source`, `total_pages`, `creation_date`, `author`, `format`, etc.
- Other loaders exist for Excel, DB, AWS S3, etc. — see [LangChain Docs](https://python.langchain.com/docs/integrations/document_loaders/)

---

## 2. Full RAG Pipeline Overview

```
Raw Files (PDFs)
      │
      ▼
[Data Ingestion] ── Load all PDFs → Document objects
      │
      ▼
[Chunking] ── Split Documents → Smaller Chunks
      │
      ▼
[Embedding Generation] ── Text → Vectors (384-dim)
      │
      ▼
[Vector Store] ── Store vectors in ChromaDB (persisted to disk)
      │
      ▼
[RAG Retriever] ── Query → Embed → Similarity Search → Context
      │
      ▼
[LLM + Prompt] ── Context + Query → Augmented Generation → Answer
```

> In this session: Steps 1–5 were fully implemented. Step 6 (LLM) is in the next video.

---

## 3. Step 1 — Data Ingestion

### Goal
Read all PDF files from a directory, load their content + metadata into `Document` objects.

### Libraries Used
```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### `process_all_pdfs()` function
```python
def process_all_pdfs(pdf_directory):
    pdf_files = Path(pdf_directory).glob("*.pdf")   # get all PDF paths

    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()                    # returns list of Document objects

        # Enrich metadata
        for doc in documents:
            doc.metadata["source_file"] = str(pdf_file)
            doc.metadata["file_type"] = "pdf"
            # Add any custom metadata keys here

        all_documents.extend(documents)

    return all_documents
```

### Result
```
Found 4 PDFs:
  - attention.pdf       → 15 pages
  - embedding.pdf       → 27 pages
  - object_detection.pdf → 21 pages
  - proposal.pdf        → 1 page
Total: 64 Document objects (one per page)
```

> **Note:** `type(all_documents[0])` → `<class 'langchain_core.documents.base.Document'>`

---

## 4. Step 2 — Chunking

### Why Chunk?
- LLMs have context window limits — you can't dump 27 pages into a prompt
- Smaller chunks = more precise retrieval
- Overlap ensures context isn't lost at chunk boundaries

### `split_documents()` function
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[" ", "\n", "\n\n"]   # priority order for splitting
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

### Key Parameters
| Parameter | Value | Meaning |
|---|---|---|
| `chunk_size` | 1000 | Max characters per chunk |
| `chunk_overlap` | 200 | Characters shared between adjacent chunks |
| `separators` | `[" ", "\n", "\n\n"]` | Try to split on these, in order |

### Result
```
64 documents → 359 chunks
```

> **Separators explained:**  
> `" "` = space, `"\n"` = newline, `"\n\n"` = paragraph break (double newline — the third separator the instructor asked the audience to guess!)

---

## 5. Step 3 — Embeddings

### Concept
Convert text (string) → dense vector (array of floats) so we can do math-based similarity search.

### Model Used
- **`all-MiniLM-L6-v2`** from HuggingFace (via `sentence-transformers`)
- Output: **384-dimensional** vector per text chunk
- Fully open-source, runs locally

### Libraries
```python
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
```

### `EmbeddingManager` class
```python
class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformers."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Protected method — loads the HuggingFace model."""
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {dim}")   # → 384

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Convert a list of strings to a numpy array of embeddings."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
```

### Usage
```python
embedding_manager = EmbeddingManager()  # loads model on init

# Extract text from chunks
texts = [doc.page_content for doc in chunks]

# Generate embeddings
embeddings = embedding_manager.generate_embeddings(texts)
# Shape: (359, 384)
```

> **Why `_load_model` (underscore prefix)?**  
> Convention for a "protected" method — signals it's internal to the class and not meant to be called directly from outside.

---

## 6. Step 4 — Vector Store

### Concept
A database optimized to store and search vectors. Instead of SQL queries, you do **similarity search** — "find the N vectors closest to this query vector."

### Tool Used: **ChromaDB** (open-source, runs locally)

### `VectorStore` class
```python
class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Create ChromaDB client and collection."""
        os.makedirs(self.persist_directory, exist_ok=True)

        # Persistent client = saves to disk
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings"}
        )
        print(f"Collection '{self.collection_name}' ready. Docs: {self.collection.count()}")

    def add_documents(self, documents, embeddings):
        """Add document chunks + their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        ids, metadatas, doc_texts, embedding_list = [], [], [], []

        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())        # unique ID for each record
            ids.append(doc_id)

            metadata = doc.metadata.copy()
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            doc_texts.append(doc.page_content)
            embedding_list.append(embedding.tolist())   # ChromaDB needs plain lists

        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            metadatas=metadatas,
            documents=doc_texts
        )
        print(f"Added {len(documents)} documents. Total in collection: {self.collection.count()}")
```

### Usage
```python
vector_store = VectorStore()
vector_store.add_documents(chunks, embeddings)
# → Total document in collection: 359
```

### Why Persist?
- `PersistentClient` saves the vector DB to disk (`./data/vector_store/`)
- On next run, you can reload without re-embedding all documents
- Massive time + compute saving for large document sets

---

## 7. Step 5 — RAG Retriever

### Concept
Given a natural language query:
1. Embed the query → query vector
2. Search vector store for the K most similar chunk vectors
3. Return those chunks as **context**

### `RAGRetriever` class
```python
class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Natural language search query
            top_k: Number of top results to return
            threshold: Minimum similarity score (0.0 = return all)

        Returns:
            List of dicts with 'content', 'metadata', 'similarity_score'
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_manager.generate_embeddings([query])
        # Shape: (1, 384)

        # Step 2: Query the vector store
        results = self.vector_store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        # results keys: 'ids', 'documents', 'metadatas', 'distances'

        # Step 3: Filter by threshold and format output
        retrieved_docs = []
        for doc_id, doc_text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity_score = 1 - distance   # ChromaDB returns distance, not similarity
            if similarity_score >= threshold:
                retrieved_docs.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": metadata,
                    "similarity_score": similarity_score
                })

        return retrieved_docs
```

> **Distance vs Similarity:**  
> ChromaDB returns a **distance** (lower = more similar).  
> `similarity = 1 - distance` converts it to an intuitive score (higher = more similar).

### Usage
```python
rag_retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)

results = rag_retriever.retrieve("What is attention is all you need?")
# → Returns list of relevant chunks with content + metadata + similarity_score
```

### Sample Output
```python
[
  {
    "content": "Attention function can be described as a mapping a query and a set of...",
    "metadata": { "source_file": "attention.pdf", "page": 3, "author": "..." },
    "similarity_score": 0.87
  },
  ...
]
```

---

## 8. Step 6 — Augmented Generation (Preview for Next Video)

The retriever gives us **context**. Next step = feed it to an LLM.

```
User Query
    │
    ├──► Embed Query ──► Search Vector Store ──► Retrieve Context
    │                                                  │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
                    Prompt = Context + Query
                              │
                              ▼
                           LLM
                              │
                              ▼
                       Final Answer
```

- **Retrieval** = get relevant chunks from vector DB
- **Augmentation** = combine context + prompt template
- **Generation** = LLM produces a grounded, factual answer

> This is what makes RAG better than plain LLM — the answer is grounded in *your* documents, not just training data.

---

## 9. Key Code Patterns

### Full Pipeline (calling order)
```python
# 1. Ingest
all_docs = process_all_pdfs("./data/pdf/")

# 2. Chunk
chunks = split_documents(all_docs, chunk_size=1000, chunk_overlap=200)

# 3. Embed
embedding_manager = EmbeddingManager()
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)

# 4. Store
vector_store = VectorStore()
vector_store.add_documents(chunks, embeddings)

# 5. Retrieve
rag_retriever = RAGRetriever(vector_store, embedding_manager)
results = rag_retriever.retrieve("your question here", top_k=5)
```

### Install dependencies
```bash
pip install langchain langchain-community pypdf sentence-transformers chromadb scikit-learn
# or if using requirements.txt:
pip install -r requirements.txt
```

---

## 10. Quick Reference Cheat Sheet

| Concept | Tool/Class | Key Method |
|---|---|---|
| Load PDFs | `PyPDFLoader` | `.load()` |
| Split text | `RecursiveCharacterTextSplitter` | `.split_documents()` |
| Generate embeddings | `EmbeddingManager` | `.generate_embeddings(texts)` |
| Store vectors | `VectorStore` (ChromaDB) | `.add_documents(docs, embeddings)` |
| Retrieve context | `RAGRetriever` | `.retrieve(query, top_k)` |

| Parameter | Default | Purpose |
|---|---|---|
| `chunk_size` | 1000 | Max chars per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `embedding_dim` | 384 | Dimensions for all-MiniLM-L6-v2 |
| `top_k` | 5 | Number of results to retrieve |
| `threshold` | 0.0 | Min similarity score to include |

### Vocabulary
- **Document** — LangChain's data structure: `{ page_content, metadata }`
- **Chunk** — smaller piece of a Document after splitting
- **Embedding** — vector representation of text (array of floats)
- **Vector Store** — DB optimized for similarity search on vectors
- **Collection** — named group of vectors inside ChromaDB
- **Retrieval** — finding chunks similar to a query vector
- **Augmentation** — combining retrieved context with a prompt for the LLM
- **Persistent Client** — ChromaDB mode that saves vectors to disk
- **UUID** — unique ID generated for each record inserted into the vector store

---

*Next up: LLM Integration — connecting the retrieved context to an LLM to generate the final answer.*