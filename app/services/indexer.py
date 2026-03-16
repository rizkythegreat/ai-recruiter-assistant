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
from datetime import datetime

class IndexerService:
    def __init__(self):
        # 1. Inisialisasi Koneksi MongoDB
        self.db_name = "recruiter_db"
        self.collection_name = "cv_embeddings"

        self.mongo_client = MongoClient(os.getenv("MONGODB_URI"))
        self.vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self.mongo_client,
            db_name=self.db_name,
            collection_name=self.collection_name,
            vector_index_name="vector_index",
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
    
    def list_indexed_files(self, user_id: str = "default") -> List[dict]:
        """
        Mengambil daftar nama file unik yang sudah terindeks di MongoDB untuk user tertentu.
        """
        collection = self.mongo_client[self.db_name][self.collection_name]

        pipeline = [
            {
                "$match": {"metadata.user_id": user_id}
            },
            {
                "$group": {
                    "_id": "$metadata.file_name",
                    "upload_date": {"$first": "$metadata.upload_date"},
                    "status": {"$first": "$metadata.status"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "file_name": "$_id",
                    "upload_date": 1,
                    "status": 1
                }
            }
        ]
        return list(collection.aggregate(pipeline))

    def delete_by_filename(self, filename: str, user_id: str = "default"):
        """
        Menghapus semua chunks dokumen yang memiliki file_name dan user_id tertentu.
        """
        collection = self.mongo_client[self.db_name][self.collection_name]
        result = collection.delete_many({
            "metadata.file_name": filename,
            "metadata.user_id": user_id
        })
        return result.deleted_count

    def save_rank_history(self, job_title: str, jd_text: str, results: List[dict], user_id: str = "default"):
        """
        Menyimpan hasil ranking ke koleksi history terpisah.
        """
        history_coll = self.mongo_client[self.db_name]["rank_history"]
        log_entry = {
            "user_id": user_id,
            "job_title": job_title,
            "job_description": jd_text,
            "results": results,
            "created_at": datetime.utcnow()
        }
        return history_coll.insert_one(log_entry).inserted_id

    def get_rank_history(self, user_id: str = "default") -> List[dict]:
        """
        Mengambil semua history ranking dari database untuk user tertentu (terbaru dulu).
        """
        history_coll = self.mongo_client[self.db_name]["rank_history"]
        return list(history_coll.find({"user_id": user_id}).sort("created_at", -1))
    
    def get_cached_embeddings(self, content_hash: str):
        """
        Cek apakah embedding untuk file dengan hash tertentu sudah ada.
        Return nodes jika ada, None jika tidak ada.
        """
        collection = self.mongo_client[self.db_name][self.collection_name]

        cached_nodes = list(collection.find({"metadata.content_hash": content_hash}))

        return cached_nodes if cached_nodes else None
    
    def clone_embeddings_for_user(
        self,
        content_hash: str,
        new_filename: str,
        new_user_id: str
    ) -> int:
        """
        Duplicate embeddings yang sudah ada untuk user baru.
        Hanya update metadata(user_id, file_name, upload_date).
        Embedding vector tetap sama (hemat token).
        """
        collection = self.mongo_client[self.db_name][self.collection_name]
        cached_nodes = list(collection.find({"metadata.content_hash": content_hash}))

        if not cached_nodes:
            return 0
        
        new_nodes = []
        current_time = datetime.utcnow().isoformat()
        for node in cached_nodes:
            new_node = node.copy()
            new_node.pop("_id", None)
            new_node["metadata"]["user_id"] = new_user_id
            new_node["metadata"]["file_name"] = new_filename
            new_node["metadata"]["upload_date"] = current_time
            new_node["metadata"]["cloned_from_cache"] = True

            new_nodes.append(new_node)
        if new_nodes:
            result = collection.insert_many(new_nodes)
            return len(result.inserted_ids)
        
        return 0