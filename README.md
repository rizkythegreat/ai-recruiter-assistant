# AI Recruiter Assistant (Advanced RAG System)

An intelligent CV Summarizer and Ranking system built with **FastAPI**, **LlamaIndex**, and **Google Gemini 1.5 Flash**. This system uses an Advanced RAG (Retrieval-Augmented Generation) pipeline to parse complex resumes, store them in **MongoDB Atlas Vector Search**, and provide intelligent candidate matching based on Job Descriptions.

## 🚀 Key Features

- **Hybrid Search (Vector + BM25):** Combines semantic meaning with keyword matching for high-precision retrieval.
- **Reciprocal Rank Fusion (RRF):** Standardizes search results from multiple retrieval systems.
- **Cross-Encoder Reranking:** Uses a secondary AI "jury" to rank the top candidates objectively.
- **Intelligent Ranking:** Automatically compares multiple candidates to a single Job Description.
- **Cloud Vector Store:** Integrated with **MongoDB Atlas** for low RAM usage and scalability.
- **Markdown Parsing:** Uses **LlamaParse** to convert PDFs and Docx into clean markdown for better LLM understanding.

## 🛠️ Tech Stack

- **Framework:** FastAPI (Python)
- **Orchestration:** LlamaIndex
- **LLM:** Google Gemini 1.5 Flash
- **Embedding:** Gemini Embedding (`gemini-embedding-001`)
- **Vector Database:** MongoDB Atlas Vector Search
- **Parsing:** LlamaParse (Llama Cloud)

## 📁 Project Structure

```text
app/
├── api/             # API Endpoints (Upload, Analyze, Rank)
├── core/            # Config, LLM Settings, and Global Dependencies
├── services/        # Business Logic (Parser, Indexer, Retriever)
├── utils/           # Helper functions (JSON cleaning, scoring)
└── main.py          # Application entry point
data/
└── uploads/         # Temporary storage for file processing
.env                 # API Keys and DB Configuration
requirements.txt     # Python dependencies
```

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher
- MongoDB Atlas account (Free Tier)
- Google AI (Gemini) API Key
- Llama Cloud (LlamaParse) API Key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-recruiter-assistant.git
cd ai-recruiter-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
# AI Keys
GOOGLE_API_KEY=your_gemini_api_key
LLAMA_CLOUD_API_KEY=your_llamaparse_api_key

# MongoDB Connection
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority

# App Config
DEBUG=True
PORT=8000
```

### 4. Running the Application
```bash
python app/main.py
```
Access the API documentation at: `http://localhost:8000/docs`

## 🔌 API Endpoints

### `POST /api/v1/upload-cv`
Upload one or many CVs (PDF/Docx). Files are parsed, indexed to MongoDB Atlas, and then automatically cleaned from local storage.

### `POST /api/v1/analyze`
Accepts a `job_description` (Form Data). Analyzes context from the uploaded resumes and returns a JSON summary, match score, strengths, and weaknesses.

### `POST /api/v1/rank-candidates`
Accepts a `job_description` (Form Data). Performs Hybrid Search + Query Fusion + Reranking to return an ordered list of candidates compared against each other.

---
Built with ❤️ for Modern HR Teams.
