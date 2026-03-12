import os
from llama_index.core import StorageContext, load_index_from_storage

# Global variables untuk RAM
_global_vector_index = None
_global_summary_index = None

def load_index_into_memory():
    """
    Memuat kedua folder index (summary & vector) ke dalam RAM.
    """
    global _global_vector_index, _global_summary_index
    
    base_path = "./storage"
    vector_path = os.path.join(base_path, "vector")
    summary_path = os.path.join(base_path, "summary")

    # Load Vector Index
    if os.path.exists(vector_path):
        print("🔍 Loading Vector Index...")
        storage_context = StorageContext.from_defaults(persist_dir=vector_path)
        _global_vector_index = load_index_from_storage(storage_context)
    
    # Load Summary Index
    if os.path.exists(summary_path):
        print("📑 Loading Summary Index...")
        storage_context = StorageContext.from_defaults(persist_dir=summary_path)
        _global_summary_index = load_index_from_storage(storage_context)

def get_vector_index():
    return _global_vector_index

def get_summary_index():
    return _global_summary_index