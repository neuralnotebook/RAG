# RAG (Retrieval Augmented Generation) — Part 1

> Covers RAG fundamentals, the two-pipeline architecture, document structure, and data loaders (Text, PDF, Directory).

---

## 1. What is RAG?

**RAG = Retrieval Augmented Generation**

> "The process of optimizing the output of a large language model so that it references an authoritative knowledge base **outside** of its training data before generating a response."

RAG extends LLM capabilities to specific domains or internal knowledge bases **without retraining the model** — making it cost-effective and always up-to-date.

### Problems RAG Solves

**Problem 1 — Hallucination**

LLMs have a training cutoff date. If you ask about events after that date, the LLM will generate a confident-sounding but **fabricated answer** (hallucination) rather than admitting it doesn't know.

**Problem 2 — Private/Domain-Specific Data**

Your company's HR policies, finance docs, internal SOPs — none of this is in the LLM's training data. You need a way to give the LLM access to this data at query time.

**Why not just fine-tune?**

Fine-tuning is expensive, slow, and needs to be redone every time your data updates. RAG is the cheaper, dynamic alternative.

> **Real-world example:** Perplexity.ai is essentially a production RAG application — it retrieves live web results, combines them with an LLM, and generates cited answers.

---

## 2. The Two Pipelines

RAG has two distinct pipelines that work together:

```
┌─────────────────────────────────────────────────────┐
│              DATA INJECTION PIPELINE                │
│                                                     │
│  Raw Data → Parse → Chunk → Embed → Vector DB       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              RETRIEVAL PIPELINE                     │
│                                                     │
│  User Query → Embed → Similarity Search → Context  │
│            → Prompt + Context → LLM → Output       │
└─────────────────────────────────────────────────────┘
```

### Pipeline 1 — Data Injection

Runs **once** (or whenever your data updates). Prepares your knowledge base.

| Step | What Happens |
|---|---|
| **Data Injection** | Load raw files (PDF, HTML, CSV, SQL, Excel, unstructured) |
| **Data Parsing** | Read and extract text content from each file |
| **Chunking** | Split large documents into smaller pieces |
| **Embedding** | Convert each chunk to a vector (numerical representation) |
| **Vector DB** | Store all vectors for similarity search |

### Pipeline 2 — Retrieval (Query Time)

Runs **every time** a user asks a question.

| Step | What Happens |
|---|---|
| **User Query** | User asks a question |
| **Embed Query** | Convert the query to a vector (same embedding model as data injection) |
| **Similarity Search** | Find chunks in Vector DB most similar to the query vector |
| **Context** | Retrieved chunks become the context |
| **Augmentation** | Context + prompt sent to LLM |
| **Generation** | LLM generates an answer grounded in the retrieved context |

> **RAG = R**etrieval + **A**ugmentation + **G**eneration — each word maps to a step in the retrieval pipeline.

---

## 3. Project Setup

```bash
uv init yt-rag
uv venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
```

### `requirements.txt`

```
langchain
langchain-core
langchain-community
langchain-openai
langchain-google-genai
pypdf
pymupdf
python-dotenv
ipykernel
```

```bash
uv add -r requirements.txt
uv add ipykernel
```

---

## 4. Document Structure

Before building the pipeline, you need to understand the **Document** data structure — it's the standard format everything gets converted into inside LangChain.

### The Document Object

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is the main text content used to build the RAG pipeline.",
    metadata={
        "source": "example.txt",
        "num_pages": 1,
        "author": "Krishna",
        "date_created": "2025-01-01"
    }
)

print(doc)
# page_content='This is the main text...' metadata={'source': 'example.txt', ...}
```

### Two Core Components

| Field | Purpose |
|---|---|
| `page_content` | The actual text content — what gets embedded and searched |
| `metadata` | Additional info: source file, page count, author, timestamps, etc. |

### Why Metadata Matters

When doing similarity search you can **filter by metadata**:

```python
# Example: only search within documents by a specific author
results = vector_db.similarity_search(
    query="leave policy",
    filter={"author": "HR Team"}
)
```

This makes retrieval much more precise — especially in large knowledge bases with many different document types.

> All LangChain document loaders output data in this same `Document` format — regardless of input file type.

---

## 5. Document Loaders

LangChain provides loaders that read various file types and automatically return a **list of `Document` objects**.

### 5.1 TextLoader — Read `.txt` files

```python
from langchain.document_loaders import TextLoader
# or
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "../data/text_files/python_intro.txt",
    encoding="utf-8"
)

documents = loader.load()
print(documents)

# Output:
# [Document(
#     page_content="Python is a general-purpose...",
#     metadata={"source": "../data/text_files/python_intro.txt"}
# )]
```

### 5.2 DirectoryLoader — Read all files in a folder

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "../data/text_files/",
    glob="*.txt",                  # file pattern to match
    loader_cls=TextLoader,         # which loader to use per file
    show_progress=False,
    use_multithreading=True
)

documents = loader.load()
print(f"Loaded {len(documents)} documents")
# → Loaded 2 documents  (one Document per .txt file)
```

### 5.3 PyMuPDFLoader — Read `.pdf` files

Two options available: `PyPDFLoader` and `PyMuPDFLoader`. **PyMuPDF is preferred** — it extracts richer metadata.

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("../data/pdf/attention.pdf")
documents = loader.load()

print(type(documents[0]))    # <class 'langchain_core.documents.base.Document'>
print(documents[0].metadata)
# {
#   'source': '../data/pdf/attention.pdf',
#   'file_path': '../data/pdf/attention.pdf',
#   'total_pages': 15,
#   'format': 'PDF 1.4',
#   'creation_date': '2024-01-01',
#   'author': 'Vaswani et al.'
# }
print(documents[0].page_content[:200])
# → "Attention Is All You Need..."
```

### 5.4 DirectoryLoader with PDFs

```python
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader(
    "../data/pdf/",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)

pdf_documents = loader.load()
print(f"Loaded {len(pdf_documents)} PDF pages")
# Each page of a PDF becomes a separate Document object
```

### Loader Comparison

| Loader | File Type | Key Metadata Extracted |
|---|---|---|
| `TextLoader` | `.txt` | source path |
| `CSVLoader` | `.csv` | source, row number |
| `PyPDFLoader` | `.pdf` | source, page number |
| `PyMuPDFLoader` | `.pdf` | source, total pages, format, creation date, author |
| `DirectoryLoader` | any (via `loader_cls`) | depends on inner loader |
| `WebBaseLoader` | URLs | source URL |

> For PDFs, always prefer `PyMuPDFLoader` over `PyPDFLoader` — the metadata is richer and parsing is more accurate.

---

## 6. Manual Document Creation

You can also create `Document` objects manually — useful for testing or custom data sources:

```python
from langchain_core.documents import Document

# Create documents manually
docs = [
    Document(
        page_content="Python is a high-level, general-purpose programming language.",
        metadata={"source": "manual", "topic": "python", "author": "Krishna"}
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications.",
        metadata={"source": "manual", "topic": "langchain", "author": "Krishna"}
    )
]

print(type(docs[0]))    # <class 'langchain_core.documents.base.Document'>
```

---

## 7. What's Next — Completing the Pipeline

So far we've covered **Data Injection** (loading + document structure). The remaining steps:

```
✅ Data Injection  →  Load raw files into Document objects
⬜ Chunking        →  Split large Documents into smaller chunks
⬜ Embeddings      →  Convert chunks to vectors (text → numbers)
⬜ Vector DB       →  Store vectors for similarity search
⬜ Retrieval       →  Query the vector DB, get context
⬜ Augmentation    →  Context + prompt → LLM
⬜ Generation      →  LLM outputs the final answer
```

### Chunking Strategies (upcoming)
- Fixed-size chunking
- Recursive character splitting
- Semantic chunking
- Context-aware chunking

### Embedding Models (upcoming)
- OpenAI embeddings (`text-embedding-3-small`)
- Google Gemini embeddings
- Open-source HuggingFace embeddings (free)

### Vector Databases (upcoming)
- Chroma (local, for development)
- FAISS (local, fast)
- Pinecone / Weaviate (cloud, production)

---

## 8. Quick Reference

```python
# Document structure
from langchain_core.documents import Document
doc = Document(page_content="...", metadata={"source": "...", "author": "..."})

# Text loader
from langchain.document_loaders import TextLoader
docs = TextLoader("file.txt", encoding="utf-8").load()

# Directory loader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
docs = DirectoryLoader("./data/", glob="*.txt", loader_cls=TextLoader).load()

# PDF loader (preferred)
from langchain_community.document_loaders import PyMuPDFLoader
docs = PyMuPDFLoader("file.pdf").load()

# Check document type
print(type(docs[0]))       # langchain_core.documents.base.Document
print(docs[0].page_content)
print(docs[0].metadata)
```