import json
import re
from typing import Dict, Any

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
