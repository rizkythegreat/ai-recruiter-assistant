from typing import List, Dict
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

class RetrieverService:
    """
    Retriever service implementing advanced hybrid (RRF) and fusion strategies.
    Combining semantic (Vector) and keyword-based (BM25) search with Reranking.
    """
    
    def __init__(self, vector_index: VectorStoreIndex):
        self.vector_index = vector_index
        # Tingkatkan k agar mencakup lebih banyak kandidat agar ranking lebih akurat
        self.vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=40)
        
        # Initialize BM25 on the nodes of the vector index
        nodes = list(self.vector_index.docstore.docs.values())
        if nodes:
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=40)
        else:
            # Fallback if no nodes are available yet
            self.bm25_retriever = None
        
        # Cross-Encoder Reranker for objective "jury" scoring
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-Minilm-L-2-v2", 
            top_n=20
        )

    def _reciprocal_rank_fusion(self, results_list: List[List[NodeWithScore]], k: int = 60) -> List[NodeWithScore]:
        """
        Standardizes ranking using RRF to combine disparate scoring systems (Vector 0-1 vs BM25 >1).
        """
        fused_scores = {}
        node_map = {}
        
        for results in results_list:
            for rank, node_with_score in enumerate(results):
                node_id = node_with_score.node.node_id
                node_map[node_id] = node_with_score.node
                
                if node_id not in fused_scores:
                    fused_scores[node_id] = 0.0
                fused_scores[node_id] += 1.0 / (k + rank + 1)
        
        # Convert back to NodeWithScore list
        fused_results = [
            NodeWithScore(node=node_map[node_id], score=score)
            for node_id, score in fused_scores.items()
        ]
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    async def hybrid_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        Executes a hybrid retrieval combining Vector and BM25 results using RRF.
        """
        vector_results = await self.vector_retriever.aretrieve(query_str)
        
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query_str)
            # Apply Reciprocal Rank Fusion
            return self._reciprocal_rank_fusion([vector_results, bm25_results])
        
        return vector_results

    async def advanced_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        Full RAG pipeline: Hybrid (RRF) -> Reranking.
        """
        # 1. Initial Hybrid Retrieval
        initial_results = await self.hybrid_retrieve(query_str)
        
        # 2. Reranking (The objective "Jury")
        query_bundle = QueryBundle(query_str)
        reranked_nodes = self.reranker.postprocess_nodes(initial_results, query_bundle)
        
        return reranked_nodes

    async def query_fusion_retrive(self, query_str: str) -> List[NodeWithScore]:
        """
        Generates sub-queries and aggregates using Hybrid RRF.
        """
        sub_queries = [
            f"Specific technical skills, coding languages, and tools required for: {query_str}",
            f"Professional work experience, roles, and achievements matching: {query_str}",
            f"Educational background, degrees, and certifications for: {query_str}"
        ]
        
        all_sub_results = []
        for q in sub_queries:
            sub_results = await self.hybrid_retrieve(q)
            all_sub_results.append(sub_results)
        
        # Fuse all sub-query results together
        fused_results = self._reciprocal_rank_fusion(all_sub_results)
        
        # Final rerank on the fused results
        query_bundle = QueryBundle(query_str)
        return self.reranker.postprocess_nodes(fused_results, query_bundle)
