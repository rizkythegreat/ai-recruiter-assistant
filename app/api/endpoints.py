import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends
from typing import List
from llama_index.core import Settings
from app.services.parser import ParserService
from app.services.indexer import IndexerService
from app.services.retriever import RetrieverService
from app.utils.helpers import clean_json_response, calculate_match_score
from app.core.dependencies import get_vector_index, load_index_into_memory
from fastapi_limiter.depends import RateLimiter
from pyrate_limiter import Duration, Limiter, Rate

router = APIRouter()
parser_service = ParserService()
indexer_service = IndexerService()

@router.post("/upload-cv", dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 60))))])
async def upload_cv(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...),
    user_id: str = Form("default_user")
):
    """
    Endpoint: /upload-cv (POST)
    Accepts PDF/Docx files, parses them, and indexes their content.
    """
    temp_paths = []
    for file in files:
        content = await file.read()
        path = parser_service.save_temp_file(content, file.filename)
        temp_paths.append(path)
    
    # 2. Jalankan proses berat di background
    background_tasks.add_task(process_and_index_cvs, temp_paths, user_id)
    
    # 3. Langsung beri respon ke frontend agar tidak timeout
    return {"message": f"Files for user {user_id} received. Indexing is processing in the background."}

async def process_and_index_cvs(file_paths: List[str], user_id: str):
        try:
            documents = await parser_service.parse_docs(file_paths, user_id=user_id)
            if documents:
                await indexer_service.build_indices(documents)
                load_index_into_memory()
        finally:
            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)

@router.post("/analyze", dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 60))))])
async def analyze_cv(
    job_description: str = Form(...),
    user_id: str = Form("default_user")
):
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
            
        retriever = RetrieverService(vector_index, user_id=user_id)
        
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

@router.post("/rank-candidates", dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 60))))])
async def rank_candidates(
    job_title: str = Form(...),
    job_description: str = Form(...),
    user_id: str = Form("default_user")
):
    """
    Endpoint: /rank-candidates (POST)
    Uses Hybrid RRF + Query Fusion + Reranking to rank all uploaded candidates.
    """
    try:
        vector_index = get_vector_index()
        if not vector_index:
            vector_index = indexer_service.load_vector_index()

        retriever = RetrieverService(vector_index, user_id=user_id)
        
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

            indexer_service.save_rank_history(
                job_title=job_title,
                jd_text=job_description,
                results=ranking,
                user_id=user_id
            )
        
        return {"job_title": job_title, "ranking": ranking}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")

@router.get('/get-history', dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 60))))])
async def get_history(user_id: str = "default_user"):
    """
    Endpoint: /get-history (GET)
    Mengambil daftar history ranking dari database untuk user tertentu.
    """
    try:
        history = indexer_service.get_rank_history(user_id=user_id)
        # Convert ObjectId ke string agar bisa di-serialize ke JSON
        for item in history:
            item["_id"] = str(item["_id"])
        return {"history": history, "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-cv", dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 60))))])
async def list_cv(user_id: str = "default_user"):
    """
    Endpoint: /list-cv (GET)
    Mengambil daftar semua nama file CV yang sudah ada di database untuk user tertentu.
    """
    try:
        files = indexer_service.list_indexed_files(user_id=user_id)
        return {"files": files, "total": len(files), "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-cv/{filename}", dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 60))))])
async def delete_cv(filename: str, user_id: str = "default_user"):
    """
    Endpoint: /delete-cv/{filename} (DELETE)
    Menghapus data CV tertentu dari database berdasarkan filename dan user_id.
    """
    try:
        count = indexer_service.delete_by_filename(filename, user_id=user_id)
        if count == 0:
            raise HTTPException(status_code=404, detail=f"File {filename} not found for user {user_id}")
        
        # Reload RAM agar index terbaru sinkron
        load_index_into_memory()
        
        return {"message": f"Successfully deleted {filename} for user {user_id} and removed {count} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))