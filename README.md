# doc-qa - Retrieval Augmented Generation

![Go](https://img.shields.io/badge/Go-1.26-00ADD8?style=flat&logo=go)
![LangChainGo](https://img.shields.io/badge/LangChainGo-v0.1.14-blue?style=flat)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?style=flat&logo=qdrant)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-white?style=flat)
![Anthropic](https://img.shields.io/badge/Anthropic-LLM-orange?style=flat)

An intelligent **Retrieval Augmented Generation** system built in Go using [LangChainGo](https://github.com/tmc/langchaingo). Ingest documents (text, PDFs, images) into a vector database, then ask natural language questions and receive AI-generated answers grounded in your data — with source attribution and conversation memory.

---

## Features

- **Multi-format Document Ingestion** — supports `.txt`, `.md`, `.csv`, `.html`, `.pdf`, and images (`.png`, `.jpg`, `.gif`, `.webp`)
- **Intelligent Image Processing** — Tesseract OCR for text-heavy images (invoices, receipts), Llava vision model fallback for photos/diagrams
- **Hybrid PDF Handling** — text extraction for digital PDFs, automatic OCR fallback for scanned documents
- **Conversational Memory** — follow-up questions use context from previous answers via session-based memory
- **Query Rewriting** — automatically rewrites follow-up questions into standalone queries for better retrieval
- **Local-first Architecture** — embeddings and OCR run entirely on your machine via Ollama and Tesseract (no data sent externally)
- **Source Attribution** — every answer includes references to the source documents used

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INGESTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Documents ──┬── Text/MD/CSV/HTML ──→ Text Splitter ──┐                     │
│              │                                         │                     │
│              ├── PDF ──→ Text Extract ──→ Splitter ────┤                     │
│              │           (fallback: pdftoppm + OCR)    │                     │
│              │                                         ├──→ Ollama ──→ Qdrant│
│              └── Images ──→ Tesseract OCR ────────────┤    Embeddings  (DB) │
│                             (fallback: Llava Vision)   │                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUERY PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Question ──→ Query Rewrite (with history) ──→ Ollama Embeddings       │
│                                                          │                   │
│                                                          ▼                   │
│                                                    Qdrant Similarity         │
│                                                       Search                 │
│                                                          │                   │
│                                                          ▼                   │
│                                                  Top-K Context Chunks        │
│                                                          │                   │
│                                                          ▼                   │
│                                                  Anthropic LLM        │
│                                                          │                   │
│                                                          ▼                   │
│                                                  Answer + Sources            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Go 1.26 | Core application |
| Framework | LangChainGo v0.1.14 | LLM orchestration, chains, document loaders |
| Vector DB | Qdrant | Store and search document embeddings |
| Embeddings | Ollama + nomic-embed-text | Convert text to 768-dim vectors (local) |
| LLM | Anthropic LLM | Generate answers from retrieved context |
| OCR | Tesseract | Extract text from images and scanned PDFs |
| Vision | Ollama + Llava | Describe images with no readable text |
| PDF Render | Poppler (pdftoppm) | Convert scanned PDF pages to images for OCR |

---

## Project Structure

```
doc-qa/
├── main.go              # CLI entry point and command routing
├── config.go            # Configuration loading and helper utilities
├── ingest/
│   └── ingest.go        # Document ingestion pipeline (text, PDF, images, OCR)
├── query/
│   └── query.go         # RAG query engine, session memory, similarity search
├── docs/
│   └── sample.txt       # Sample document for testing
├── .env.example         # Environment variable template
├── Makefile             # Build and run shortcuts
├── go.mod               # Go module definition
├── go.sum               # Dependency checksums
└── README.md            # This file
```

---

## Prerequisites

### 1. Ollama (Local Embeddings & Vision)

```bash
# macOS
brew install ollama

# Linux/Windows — download from https://ollama.com/download
```

Start the server and pull models:

```bash
# Start Ollama (runs on http://localhost:11434)
ollama serve

# Pull the embedding model (~274MB, required)
ollama pull nomic-embed-text

# Pull the vision model (~4.7GB, only needed for image ingestion)
ollama pull llava
```

### 2. Tesseract OCR & Poppler (Optional — Image/PDF Text Extraction)

> **Note:** Tesseract is optional. Without it, image ingestion gracefully falls back to the Llava vision model. The project compiles and runs without CGO or Tesseract installed.

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows — download installers:
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler: https://github.com/oschwartz10612/poppler-windows/releases
```

### 3. Qdrant (Vector Database)

```bash
# Using Docker (recommended)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

The collection `rag-documents` is created automatically on first ingestion.

### 4. Anthropic API Key (for Query)

```bash
export ANTHROPIC_AUTH_TOKEN="your-anthropic-api-key"
```

> See `.env.example` for all available configuration options.

---

## Quick Start

```bash
# 1. Clone and navigate to the project
cd doc-qa

# 2. Start services
ollama serve &
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. Pull required models (one-time)
ollama pull nomic-embed-text

# 4. Ingest sample document
go run . ingest docs/sample.txt

# 5. Set your API key and query
export ANTHROPIC_AUTH_TOKEN="your-key"
go run . query "What are the benefits of RAG?"
```

Or using the Makefile:

```bash
make setup          # Start Qdrant via Docker
make pull-models    # Download Ollama models
make ingest-sample  # Ingest the sample document
make run-query      # Start interactive query mode
```

---

## Usage

### Ingest Documents

```bash
# Single file
go run . ingest path/to/document.txt

# Multiple files (mixed types)
go run . ingest report.pdf notes.md screenshot.png

# Entire directory (recursively finds supported files)
go run . ingest --dir ./my-documents/

# Raw text directly from CLI
go run . ingest --text "Go is a statically typed language designed at Google."
```

### Query (RAG with LLM)

```bash
# Single question — get an answer with source attribution
go run . query "What is RAG and how does it work?"

# Interactive mode — conversational session with memory
go run . query
> Question: What are the main components?
> Question: Tell me more about the first one   ← uses conversation context
> Question: exit
```

### Search (Similarity Only)

```bash
# Raw vector search — returns matching chunks without LLM generation
go run . search "vector databases"
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_AUTH_TOKEN` | *(required for query)* | Anthropic API key |
| `ANTHROPIC_BASE_URL` | Anthropic default | Custom API endpoint |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model for answer generation |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for text embeddings |
| `VISION_MODEL` | `llava` | Model for image description |

---

## How It Works

### 1. Document Ingestion

```
File → Detect Type → Load & Extract Text → Chunk (500 chars / 50 overlap) → Embed → Store in Qdrant
```

- **Text files** are split using recursive character splitting for optimal chunk boundaries
- **PDFs** attempt direct text extraction first; scanned PDFs fall back to page rendering (pdftoppm) + OCR (Tesseract)
- **Images** use Tesseract OCR for text-heavy images; fall back to Llava vision model when OCR yields < 50 characters

### 2. Query Processing

```
Question → (Rewrite with History) → Embed → Similarity Search → Top-K Chunks → LLM → Answer
```

- In interactive mode, follow-up questions are rewritten into standalone queries using conversation history
- Similarity search uses cosine distance with a 0.5 score threshold
- Retrieved chunks are passed as context to the Anthropic LLM for grounded answer generation

### 3. Similarity Search

```
Query → Embed → Cosine Similarity → Top-5 Chunks → Display
```

- Direct vector search without LLM generation — useful for debugging or verifying what's in the database

---

## Supported File Types

| Extension | Method | Notes |
|-----------|--------|-------|
| `.txt`, `.md` | Text splitter | Split into chunks directly |
| `.csv` | CSV loader | Each row becomes a document |
| `.html` | HTML parser | Extracts text content |
| `.pdf` | Text extraction + OCR fallback | Handles both digital and scanned PDFs |
| `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Tesseract OCR + Llava fallback | OCR for text; vision model for photos |

---

## Build

```bash
# Build binary (works without CGO/Tesseract — OCR falls back to vision model)
go build -o doc-qa .

# Build with Tesseract OCR support (requires Tesseract installed)
CGO_ENABLED=1 go build -o doc-qa .

# Run the binary directly
./doc-qa ingest docs/sample.txt
./doc-qa query "What is RAG?"
```

---

## Acknowledgments

- [LangChainGo](https://github.com/tmc/langchaingo) — Go port of the LangChain framework
- [Qdrant](https://qdrant.tech/) — High-performance vector database
- [Ollama](https://ollama.com/) — Run LLMs locally
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Open-source OCR engine
- [Anthropic](https://www.anthropic.com/) — LLM for answer generation
