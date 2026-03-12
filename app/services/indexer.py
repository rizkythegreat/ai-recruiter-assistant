import os
from typing import List
from llama_index.core import (
    VectorStoreIndex,
    DocumentSummaryIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from app.core.config import Config

class IndexerService:
    """
    Indexer service to manage Vector and Summary indices.
    Chunking strategy uses SentenceSplitter for optimal CV segments.
    """
    
    def __init__(self, persist_dir: str = "./storage"):
        self.persist_dir = persist_dir
        self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    async def build_indices(self, documents: List[Document]) -> VectorStoreIndex:
        vector_path = os.path.join(self.persist_dir, "vector")
        
        # 1. Cek apakah index sudah ada di disk
        if os.path.exists(vector_path):
            # Load index yang lama
            storage_context = StorageContext.from_defaults(persist_dir=vector_path)
            vector_index = load_index_from_storage(storage_context)
            
            # Tambahkan dokumen baru ke index yang lama
            for doc in documents:
                vector_index.insert(doc)
        else:
            # Jika belum ada (pertama kali), buat baru
            vector_index = VectorStoreIndex.from_documents(
                documents, 
                transformations=[self.node_parser]
            )
        
        # 2. Simpan kembali (Persist)
        os.makedirs(self.persist_dir, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=vector_path)
        
        # Update memori global (Opsional: panggil load_index_into_memory() di sini)
        
        return vector_index

    def load_vector_index(self) -> VectorStoreIndex:
        """
        Loads the pre-built Vector index from disk.
        """
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(self.persist_dir, "vector")
        )
        return load_index_from_storage(storage_context)

    def load_summary_index(self) -> DocumentSummaryIndex:
        """
        Loads the pre-built Summary index from disk.
        """
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(self.persist_dir, "summary")
        )
        return load_index_from_storage(storage_context)
