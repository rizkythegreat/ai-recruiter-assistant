import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

load_dotenv()

class Config:
    """
    Configuration class for the AI Recruiter Assistant.
    Handles environment variables and LlamaIndex settings.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    
    # LLM Settings
    MODEL_NAME = "gemini-3.1-flash-lite-preview"
    EMBED_MODEL_NAME = "gemini-embedding-001"
    
    @staticmethod
    def initialize_settings():
        """
        Initialize LlamaIndex global settings with Gemini models.
        """
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        # Configure LLM
        llm = Gemini(model_name=Config.MODEL_NAME, api_key=Config.GOOGLE_API_KEY)
        
        # Configure Embedding
        embed_model = GeminiEmbedding(model_name=Config.EMBED_MODEL_NAME, api_key=Config.GOOGLE_API_KEY)
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

# Initialize settings on import
Config.initialize_settings()
