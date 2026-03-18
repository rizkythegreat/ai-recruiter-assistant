from app.services.indexer import IndexerService
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import Config

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == Config.API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate API Key"
    )

# Global variable agar index tetap di RAM selama aplikasi berjalan
_global_vector_index = None

def load_index_into_memory():
    """
    Create MongoDB connection index to internal memory apps.
    """
    global _global_vector_index
    try:
        print("🔍 Connecting to MongoDB Atlas Vector Index...")
        indexer = IndexerService()
        _global_vector_index = indexer.load_vector_index()
        print("✅ MongoDB Index Connected!")
    except Exception as e:
        print(f"❌ Failed to load MongoDB index: {e}")

def get_vector_index():
    return _global_vector_index