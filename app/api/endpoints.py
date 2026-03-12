from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Dict, Any
import json
from llama_index.core import Settings
from app.services.parser import ParserService
from app.services.indexer import IndexerService
from app.services.retriever import RetrieverService
from app.utils.helpers import clean_json_response, calculate_match_score
from app.core.dependencies import get_vector_index, load_index_into_memory

router = APIRouter()
parser_service = ParserService()
indexer_service = IndexerService()

@router.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    """
    Endpoint: /upload-cv (POST)
    Accepts PDF/Docx files, parses them, and indexes their content.
    """
    try:
        saved_paths = []
        for file in files:
            content = await file.read()
            path = parser_service.save_temp_file(content, file.filename)
            saved_paths.append(path)
        
        documents = await parser_service.parse_docs(saved_paths)
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to parse documents.")
            
        await indexer_service.build_indices(documents)
        
        # Reload index into RAM immediately
        load_index_into_memory()

        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return {"message": f"Successfully indexed {len(files)} CVs."}
        
    except Exception as e:
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_cv(job_description: str = Form(...)):
    """
    Endpoint: /analyze (POST)
    Retreives relevant chunks using Hybrid RRF + Reranking and analyzes.
    """
    try:
        # Load index from memory (using dependency)
        vector_index = get_vector_index()
        if not vector_index:
            # Fallback if not loaded
            vector_index = indexer_service.load_vector_index()
            
        retriever = RetrieverService(vector_index)
        
        # Perform Fusion Retrieval with Reranking
        relevant_nodes = await retriever.advanced_retrieve(job_description)
        context_str = "\n".join([node.node.get_content() for node in relevant_nodes])
        
        prompt = f"""
        Analyze the candidate CV details against the Job Description (JD).
        JD: {job_description}
        Context: {context_str}
        
        Output REQUIREMENT: Return ONLY a valid JSON object. No preamble.
        Format:
        {{
          "summary": "Concise summary",
          "match_score": 0-100,
          "strengths": ["s1", "s2"],
          "weaknesses": ["w1", "w2"]
        }}
        """
        
        response = await Settings.llm.acomplete(prompt)
        analysis = clean_json_response(str(response))
        analysis["match_score"] = calculate_match_score(analysis.get("match_score", 0))
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/rank-candidates")
async def rank_candidates(job_description: str = Form(...)):
    """
    Endpoint: /rank-candidates (POST)
    Uses Hybrid RRF + Query Fusion + Reranking to rank all uploaded candidates.
    """
    try:
        vector_index = get_vector_index()
        if not vector_index:
            vector_index = indexer_service.load_vector_index()

        retriever = RetrieverService(vector_index)
        
        # 1. Broad retrieval across multiple facets of the JD
        relevant_nodes = await retriever.query_fusion_retrive(job_description)
        
        # 2. Group findings by candidate (file_name)
        candidates_map = {}
        for node in relevant_nodes:
            fname = node.node.metadata.get("file_name", "Unknown")
            if fname not in candidates_map:
                candidates_map[fname] = []
            candidates_map[fname].append(node.node.get_content())

        # 3. Formulate comparative prompt
        candidates_context = ""
        for name, contents in candidates_map.items():
            combined_text = "\n".join(contents)[:3000] # Safe context window per candidate
            candidates_context += f"--- Candidate: {name} ---\n{combined_text}\n\n"

        prompt = f"""
        Role: Senior Technical Recruiter.
        JD: {job_description}

        Candidates Data:
        {candidates_context}

        Task: Rank candidates by suitability and extract metadata. 
        Output REQUIREMENT: Return ONLY a valid JSON list of objects. No preamble. 
        Format per candidate:
        [
          {{
            "candidate": "filename.pdf",
            "score": 0-100,
            "metadata": {{
                "years_of_experience": number,
                "top_skills": ["skill1", "skill2"],
                "location": "string"
            }},
            "analysis": {{
                "reason": "Brief comparison",
                "suitability_tag": "Highly Recommended/Medium Match/Low Match"
            }}
          }}
        ]
        """

        response = await Settings.llm.acomplete(prompt)
        ranking = clean_json_response(str(response))
        
        if isinstance(ranking, list):
            ranking.sort(key=lambda x: x.get("score", 0), reverse=True)
            for i, item in enumerate(ranking):
                item["rank"] = i + 1
        
        return {"ranking": ranking}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")

@router.get("/list-cv")
async def list_cv():
    """
    Endpoint: /list-cv (GET)
    Mengambil daftar semua nama file CV yang sudah ada di database.
    """
    try:
        files = indexer_service.list_indexed_files()
        return {"files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-cv/{filename}")
async def delete_cv(filename: str):
    """
    Endpoint: /delete-cv/{filename} (DELETE)
    Menghapus data CV tertentu dari database.
    """
    try:
        count = indexer_service.delete_by_filename(filename)
        if count == 0:
            raise HTTPException(status_code=404, detail="File not found in database.")
        
        # Reload RAM agar index terbaru sinkron
        load_index_into_memory()
        
        return {"message": f"Successfully deleted {filename} and removed {count} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))