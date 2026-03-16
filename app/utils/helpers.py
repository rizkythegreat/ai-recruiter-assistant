import json
import os
import re
from typing import Dict, Any, List, Optional

def clean_json_response(raw_response: str) -> Dict[str, Any]:
    """
    Cleans and parses a JSON response from the LLM.
    Handles potential markdown code blocks or extra whitespace.
    
    Args:
        raw_response (str): The raw string response from the LLM.
        
    Returns:
        Dict[str, Any]: The parsed JSON object.
    """
    # Remove potential markdown JSON code block markers
    clean_str = re.sub(r'```json\s*|\s*```', '', raw_response).strip()
    
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # Fallback: find the first { and last } to isolate potential JSON
        start_idx = clean_str.find('{')
        end_idx = clean_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(clean_str[start_idx:end_idx+1])
            except json.JSONDecodeError:
                pass
        
        return {"error": "Failed to parse JSON response", "raw_content": raw_response}

def calculate_match_score(score: float) -> int:
    """
    Normalizes a matching score to an integer between 0 and 100.
    
    Args:
        score (float): The input score.
        
    Returns:
        int: Normalized score.
    """
    try:
        return int(max(0, min(100, score)))
    except (ValueError, TypeError):
        return 0

def load_preset_result(job_title: str, candidate_files: List[str]) -> Optional[dict]:
    """
    Load preset ranking dari file JSON jika ada dan kandidat cocok.
    """
    # Normalize job title jadi filename (improved)
    # 1. Remove parentheses and content inside
    normalized = re.sub(r'\([^)]*\)', '', job_title)
    # 2. Replace / and extra spaces
    normalized = normalized.replace('/', ' ')
    # 3. Replace multiple spaces with single dash
    normalized = re.sub(r'\s+', '-', normalized.strip())
    # 4. Remove multiple consecutive dashes
    normalized = re.sub(r'-+', '-', normalized)
    # 5. Lowercase
    preset_filename = normalized.lower() + '.json'
    
    preset_path = os.path.join('data', 'presets', preset_filename)
    
    if not os.path.exists(preset_path):
        return None
    
    try:
        with open(preset_path, 'r') as f:
            preset_data = json.load(f)
        
        # Cek apakah kandidat cocok (harus exact match)
        preset_files = sorted(preset_data.get('candidate_files', []))
        current_files = sorted(candidate_files)
        
        if preset_files == current_files:
            return preset_data
        
        return None
    
    except Exception as e:
        print(f"Error loading preset: {e}")
        return None