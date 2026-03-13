import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.core.config import Config
from app.core.dependencies import load_index_into_memory

# Ensure initial config is loaded
Config.initialize_settings()

app = FastAPI(
    title="AI Recruiter Assistant",
    description="Advanced RAG system for CV analysis and candidate scoring.",
    version="1.0.0"
)

# Load indices into RAM at startup for better performance
@app.on_event("startup")
async def startup_event():
    print("Initializing Indices into memory...")
    load_index_into_memory()

# CORS Middleware for potential frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"status": "AI Recruiter Assistant is Running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
