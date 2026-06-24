# RAG Pipeline — Part 2: LLM Integration & Modular Architecture
> Source: YouTube transcript (continuation) | Picks up from: Query Retrieval → LLM output

---

## Table of Contents
1. [Quick Recap — Where Part 1 Ended](#1-quick-recap)
2. [Step 6 — LLM Integration (Simple RAG)](#2-step-6--llm-integration-simple-rag)
3. [Enhanced RAG Pipeline](#3-enhanced-rag-pipeline)
4. [Advanced RAG Pipeline (Streaming + Citations + History)](#4-advanced-rag-pipeline)
5. [Modular Project Structure](#5-modular-project-structure)
6. [Module 1 — `data_loader.py`](#6-module-1--data_loaderpy)
7. [Module 2 — `embedding.py` (EmbeddingPipeline class)](#7-module-2--embeddingpy)
8. [Module 3 — `vector_store.py` (FAISS)](#8-module-3--vector_storepy)
9. [Module 4 — `search.py` (RAGSearch + LLM)](#9-module-4--searchpy)
10. [Wiring It All Together — `app.py`](#10-wiring-it-all-together--apppy)
11. [ChromaDB vs FAISS — Quick Comparison](#11-chromadb-vs-faiss)
12. [Cheat Sheet](#12-cheat-sheet)

---

## 1. Quick Recap

By end of Part 1, the full data ingestion pipeline was complete:

```
PDF files → Document objects → Chunks → Embeddings → ChromaDB (persisted)
                                                            ↓
                                                    RAGRetriever.retrieve(query)
                                                            ↓
                                                    Relevant context chunks
```

**What was missing:** feeding that context to an LLM to produce a final answer.  
**Part 2 covers:** LLM integration + refactoring everything into a clean modular codebase.

---

## 2. Step 6 — LLM Integration (Simple RAG)

### LLM Used: Groq (fast inference, free tier available)
Model: `gemma2` | Temp: `0.1` | Max tokens: `1024`

### Setup
```bash
pip install langchain-groq python-dotenv
```

```python
# .env file
GROQ_API_KEY=your_key_here
```

```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="gemma2",
    temperature=0.1,
    max_tokens=1024
)
```

> **Tip from instructor:** For testing, paste the key directly first to confirm it works before using `os.getenv()`.

### `rag_simple()` — The Core Function

```python
def rag_simple(query: str, retriever, llm, top_k: int = 3) -> str:
    # Step 1: Retrieve relevant context from vector store
    results = retriever.retrieve(query, top_k=top_k)

    # Step 2: Combine all retrieved chunks into one context string
    if results:
        context = "\n\n".join([doc["content"] for doc in results])
    else:
        context = ""

    # Step 3: Guard — if no context found, bail early
    if not context:
        return "No relevant context found to answer the question."

    # Step 4: Build the prompt
    prompt = """Use the following context to answer the question concisely.

Context:
{context}

Question:
{query}

Answer:"""

    # Step 5: Call the LLM
    response = llm.invoke(prompt.format(context=context, query=query))

    return response.content
```

### Usage
```python
answer = rag_simple(
    query="What is attention mechanism?",
    retriever=rag_retriever,
    llm=llm
)
print(answer)
# → "Attention mechanism is a function that maps a query..."
```

### What's happening under the hood
```
User Query
   │
   ├─► retriever.retrieve(query) ──► top-K chunks from vector DB
   │                                         │
   └─────────────────────────────────────────┘
                       │
                       ▼
         prompt = context + query instructions
                       │
                       ▼
                  llm.invoke(prompt)
                       │
                       ▼
                 Final Answer
```

---

## 3. Enhanced RAG Pipeline

The simple version just returns an answer. The **enhanced version** also returns:
- Source file + page number for each retrieved chunk
- Similarity/confidence score
- Content preview (first 300 chars)
- Optionally the full context

```python
def rag_advanced(
    query: str,
    retriever,
    llm,
    top_k: int = 5,
    minimum_score: float = 0.0,
    return_context: bool = False
) -> dict:

    # Step 1: Retrieve
    results = retriever.retrieve(query, top_k=top_k, threshold=minimum_score)

    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
            "context": ""
        }

    # Step 2: Build context + extract metadata
    context_parts = []
    sources = []

    for doc in results:
        context_parts.append(doc["content"])

        sources.append({
            "source_file": doc["metadata"].get("source_file"),
            "page": doc["metadata"].get("page"),
            "similarity_score": doc["similarity_score"],
            "preview": doc["content"][:300]          # first 300 chars
        })

    context = "\n\n".join(context_parts)

    # Step 3: Calculate average confidence
    confidence = sum(doc["similarity_score"] for doc in results) / len(results)

    # Step 4: Prompt + LLM call
    prompt = """Use the following context to answer the question precisely.

Context:
{context}

Question:
{query}

Answer:"""

    response = llm.invoke(prompt.format(context=context, query=query))

    return {
        "answer": response.content,
        "sources": sources,
        "confidence": confidence,
        "context": context if return_context else ""
    }
```

### Sample Output
```python
{
  "answer": "The attention mechanism maps a query and set of key-value pairs...",
  "sources": [
    {
      "source_file": "attention.pdf",
      "page": 3,
      "similarity_score": 0.87,
      "preview": "Attention function can be described as..."
    }
  ],
  "confidence": 0.84,
  "context": "..."   # only if return_context=True
}
```

---

## 4. Advanced RAG Pipeline

A third, even richer variant that adds:
- **Streaming** — tokens stream back as they're generated
- **Citations** — inline references tied to source documents
- **Conversation history** — multi-turn context
- **Summarization** — condenses long contexts before passing to LLM

> The instructor didn't walk through this code line-by-line — it was shown as a reference. The key idea is the same pipeline, just with these features layered on top. Worth reading through on your own.

**Takeaway pattern:**
```
3 levels of RAG complexity:
  1. rag_simple       → answer only
  2. rag_advanced     → answer + sources + confidence + context preview  
  3. rag_advanced_v3  → answer + streaming + citations + history + summarization
```

---

## 5. Modular Project Structure

The notebook code gets refactored into a proper Python project:

```
project/
├── app.py                  ← entry point, wires everything together
├── data/
│   ├── pdf/                ← raw PDF files
│   └── fire_store/         ← persisted FAISS vector store
│       ├── files.index
│       └── metadata.pickle
├── src/
│   ├── __init__.py
│   ├── data_loader.py      ← load PDFs/TXT/CSV → Document objects
│   ├── embedding.py        ← chunk + embed documents
│   ├── vector_store.py     ← FAISS store (build, save, load, search)
│   └── search.py           ← RAG search + LLM integration
└── .env                    ← GROQ_API_KEY
```

> **Why modular?** Each component has a single responsibility. You can swap out the vector store (FAISS → ChromaDB) or LLM (Groq → OpenAI) without touching other files.

---

## 6. Module 1 — `data_loader.py`

Responsibility: read any supported file type → list of LangChain `Document` objects.

```python
# src/data_loader.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.document_loaders import CSVLoader  # for CSV

def load_all_documents(data_directory: str) -> list:
    """
    Load all supported files from data_directory.
    Returns: list of LangChain Document objects.
    """
    data_path = Path(data_directory)
    documents = []

    # --- PDF files ---
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())

    # --- TXT files ---
    txt_files = list(data_path.glob("**/*.txt"))
    for txt in txt_files:
        loader = TextLoader(str(txt))
        documents.extend(loader.load())

    # --- ASSIGNMENT: add CSV, SQL, etc. ---
    # csv_files = list(data_path.glob("**/*.csv"))
    # for csv in csv_files:
    #     loader = CSVLoader(str(csv))
    #     documents.extend(loader.load())

    return documents
```

> **Assignment from instructor:** extend this to support CSV, SQL, and other file types using the appropriate LangChain loaders.

---

## 7. Module 2 — `embedding.py` (EmbeddingPipeline class)

Responsibility: chunk documents + convert chunks to vector embeddings.

```python
# src/embedding.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Embedding model loaded: {model_name}")

    def chunk_documents(self, documents: list) -> list:
        """Split documents into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[" ", "\n", "\n\n"]
        )
        chunks = splitter.split_documents(documents)
        print(f"Splitted {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: list):
        """Convert chunks to numpy embeddings array."""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings   # shape: (num_chunks, 384)
```

### Usage in `app.py`
```python
from src.embedding import EmbeddingPipeline

pipeline = EmbeddingPipeline()
chunks = pipeline.chunk_documents(documents)
chunk_vectors = pipeline.embed_chunks(chunks)
```

---

## 8. Module 3 — `vector_store.py` (FAISS)

> **Note:** This module switches from ChromaDB (used in Part 1) to **FAISS** (`faiss-cpu`). Both are valid; FAISS is more commonly used in production for speed.

Responsibility: store embeddings persistently, load them back, and run similarity search.

```python
# src/vector_store.py
import faiss
import pickle
import numpy as np
from pathlib import Path
from src.embedding import EmbeddingPipeline

class FAISSVectorStore:
    def __init__(self, store_path: str = "data/fire_store"):
        self.store_path = Path(store_path)
        self.index_path = self.store_path / "files.index"
        self.metadata_path = self.store_path / "metadata.pickle"
        self.model = EmbeddingPipeline()
        self.index = None
        self.metadata = []

    def build_from_documents(self, documents: list):
        """Full pipeline: chunk → embed → index → save."""
        chunks = self.model.chunk_documents(documents)
        embeddings = self.model.embed_chunks(chunks)

        # Build FAISS index
        dim = embeddings.shape[1]           # 384
        self.index = faiss.IndexFlatL2(dim) # L2 distance
        self.index.add(embeddings.astype(np.float32))

        # Store metadata alongside
        self.metadata = [
            {"content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        self._save()

    def _save(self):
        """Persist index and metadata to disk."""
        self.store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved FAISS index + metadata to {self.store_path}")

    def load(self):
        """Load persisted index and metadata from disk."""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def query(self, query_text: str, top_k: int = 3) -> list:
        """Search for similar documents given a text query."""
        query_vec = self.model.model.encode([query_text]).astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                **self.metadata[idx],
                "distance": float(dist)
            })
        return results
```

### Key FAISS concepts
| Concept | Explanation |
|---|---|
| `IndexFlatL2` | Exact search using L2 (Euclidean) distance — simple, accurate |
| `index.add(embeddings)` | Add vectors to the index |
| `index.search(vec, k)` | Returns `(distances, indices)` for top-k results |
| `faiss.write_index` | Save index to `.index` file |
| `faiss.read_index` | Load index from `.index` file |
| `metadata.pickle` | Stores original text + metadata alongside the index |

---

## 9. Module 4 — `search.py` (RAGSearch + LLM)

Responsibility: wrap vector store query + LLM call into one clean interface.

```python
# src/search.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class RAGSearch:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="gemma2",
            temperature=0.1,
            max_tokens=1024
        )

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        # Step 1: Get relevant docs from FAISS
        results = self.vector_store.query(query, top_k=top_k)

        if not results:
            return "No relevant context found."

        # Step 2: Build context
        context = "\n\n".join([r["content"] for r in results])

        # Step 3: Prompt + LLM
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content
```

---

## 10. Wiring It All Together — `app.py`

```python
# app.py
from src.data_loader import load_all_documents
from src.vector_store import FAISSVectorStore
from src.search import RAGSearch
from pathlib import Path

if __name__ == "__main__":
    store_path = "data/fire_store"

    # Initialize vector store
    store = FAISSVectorStore(store_path=store_path)

    if Path(store_path).exists():
        # Already built — just load from disk
        store.load()
    else:
        # First run — build from documents
        documents = load_all_documents("data/")
        store.build_from_documents(documents)

    # Run a query
    searcher = RAGSearch(vector_store=store)
    answer = searcher.search_and_summarize("What is attention mechanism?")
    print(answer)
```

### First run output
```
Found 4 PDF files
Load embedding model: all-MiniLM-L6-v2
Splitted 64 documents into 359 chunks
Generating embeddings: 100%|████| 359/359
Saved FAISS index + metadata to data/fire_store
Answer: The attention mechanism maps a query and a set of...
```

### Second run output (loads from disk — much faster)
```
Loading embedding model...
Loaded FAISS index: 359 vectors
Answer: The attention mechanism maps a query and a set of...
```

> **Key win:** persistence means you only embed once. On every subsequent run, you skip the expensive embedding step entirely.

---

## 11. ChromaDB vs FAISS

| Feature | ChromaDB | FAISS |
|---|---|---|
| Interface | High-level (collections, filters) | Low-level (index, arrays) |
| Persistence | Built-in `PersistentClient` | Manual save/load with `pickle` |
| Metadata storage | Built-in | Manual (pickle file) |
| Speed | Good | Very fast (optimized C++ core) |
| Filtering | Supports metadata filters | Not natively (needs wrapper) |
| Best for | Prototypes, flexibility | Production, speed |
| Part used in | Part 1 (Jupyter notebook) | Part 2 (modular code) |

Both are **open-source** and **run locally** — no API key needed for the vector store itself.

---

## 12. Cheat Sheet

### Full Modular Pipeline Flow
```
app.py
  │
  ├─► data_loader.load_all_documents("data/")
  │         └─► PyPDFLoader + TextLoader → list[Document]
  │
  ├─► EmbeddingPipeline.chunk_documents(docs)
  │         └─► RecursiveCharacterTextSplitter → list[Document chunks]
  │
  ├─► EmbeddingPipeline.embed_chunks(chunks)
  │         └─► SentenceTransformer.encode() → np.ndarray (359, 384)
  │
  ├─► FAISSVectorStore.build_from_documents(docs)   [first run only]
  │         └─► faiss.IndexFlatL2 + pickle → saved to disk
  │
  ├─► FAISSVectorStore.load()                        [subsequent runs]
  │
  └─► RAGSearch.search_and_summarize(query)
            ├─► vector_store.query(query, top_k)
            ├─► build context string
            ├─► format prompt
            └─► llm.invoke(prompt) → answer
```

### The 3 RAG pipeline variants

| Variant | Returns | When to use |
|---|---|---|
| `rag_simple` | answer (str) | Quick prototyping |
| `rag_advanced` | answer + sources + confidence + preview | Production with attribution |
| `rag_advanced_v3` | above + streaming + citations + history + summarization | Full-featured chatbot |

### Groq LLM init (quick copy)
```python
from langchain_groq import ChatGroq
llm = ChatGroq(groq_api_key="...", model_name="gemma2", temperature=0.1, max_tokens=1024)
response = llm.invoke("your prompt here")
print(response.content)
```

---

*Next up: Further optimizations — chunking strategies, hybrid search, reranking.*