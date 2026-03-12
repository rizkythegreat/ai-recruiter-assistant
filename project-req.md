# Blueprint: AI Recruiter Assistant (Advanced RAG System)

## Context
I want to build a "CV Summarizer & AI Recruiter Assistant" using Python. The goal is to create an API that can parse complex resumes (PDF/Docx), index them using advanced retrieval strategies, and provide intelligent summaries/scoring based on a Job Description (JD).

## Tech Stack
- **Framework:** FastAPI
- **Orchestration:** LlamaIndex
- **Parser:** LlamaParse (Markdown mode)
- **LLM:** Google Gemini 1.5 Flash
- **Embedding:** Google Gemini Embedding
- **Vector Store:** ChromaDB or Qdrant
- **Retrieval:** Hybrid Search (Vector + BM25), Auto-merging, and Query Fusion.

## Project Structure
Please initialize the project with the following structure:
ai-recruiter-assistant/
├── app/
│   ├── main.py              # Entry point FastAPI
│   ├── api/                 # Endpoint routes
│   │   └── endpoints.py     # POST /upload-cv and POST /analyze
│   ├── core/                # Config LLM & Gemini
│   │   └── config.py
│   ├── services/            # Business Logic (RAG)
│   │   ├── parser.py        # LlamaParse integration
│   │   ├── indexer.py       # Vector & Summary Indexing
│   │   └── retriever.py     # Advanced Retrieval (Hybrid + Fusion)
│   └── utils/               # Helpers
│       └── helpers.py       # JSON formatting & scoring logic
├── .env                     # API Keys
└── requirements.txt         # Dependencies

## Implementation Requirements

### 1. Parser (services/parser.py)
- Integrate `LlamaParse`. 
- Set result_type to "markdown".
- Handle multiple file uploads.

### 2. Indexer (services/indexer.py)
- Implement `VectorStoreIndex` for semantic search.
- Implement `DocumentSummaryIndex` for high-level CV overview.
- Use `SentenceSplitter` for optimal chunking.

### 3. Retriever (services/retriever.py)
- Build a **Query Fusion Retriever**: Generate 3 sub-queries from the Job Description.
- Build a **Hybrid Retriever**: Combine `VectorIndexRetriever` and `BM25Retriever`.
- Implement **Auto-merging logic**: Ensure child chunks are merged back to parent context if relevancy is high.

### 4. API Endpoints (api/endpoints.py)
- `POST /upload-cv`: Accepts PDF, parses it via LlamaParse, and stores it in the vector database.
- `POST /analyze`: Accepts a Job Description string, retrieves relevant CV parts, and uses Gemini 1.5 Flash to generate:
    - Candidate Summary.
    - Skill Match Score (0-100).
    - Strengths and Weaknesses.

### 5. Formatting
- All LLM responses must be in structured JSON.
- Use asynchronous functions (`async/await`) for all API and LLM calls.

## Prompt Goal
"Create the foundational code for this entire structure. Start by generating the `requirements.txt` and `.env` template, then proceed to build each module starting from the core configuration to the final FastAPI endpoints. Ensure all code is clean, documented with JSDoc-style comments, and follows production-ready Python patterns."