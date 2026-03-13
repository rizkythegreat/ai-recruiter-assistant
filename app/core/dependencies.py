from app.services.indexer import IndexerService

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