import os
from typing import List
from pymongo import MongoClient
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from app.core.config import Config

class IndexerService:
    def __init__(self):
        # 1. Inisialisasi Koneksi MongoDB
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        self.vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self.mongo_client,
            db_name="recruiter_db",
            collection_name="cv_embeddings",
            index_name="vector_index"
        )
        self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    async def build_indices(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Upload documents dan embedding-nya langsung ke MongoDB Atlas.
        """
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Buat index (ini otomatis mengirim data ke Cloud MongoDB)
        vector_index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            transformations=[self.node_parser],
            show_progress=True
        )
        return vector_index

    def load_vector_index(self) -> VectorStoreIndex:
        """
        Gunakan Vector Store langsung dari database Cloud.
        """
        return VectorStoreIndex.from_vector_store(vector_store=self.vector_store)