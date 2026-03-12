import os
import shutil
from typing import List, Union
from llama_parse import LlamaParse
from llama_index.core import Document
from app.core.config import Config

class ParserService:
    """
    Parser service for documents such as CVs and JDs.
    Integrates LlamaParse for high-quality PDF/Docx markdown parsing.
    """
    
    def __init__(self):
        self.parser = LlamaParse(
            api_key=Config.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            num_workers=4,
            verbose=os.getenv("DEBUG", "False").lower() == "true"
        )
    
    async def parse_docs(self, file_paths: Union[str, List[str]]) -> List[Document]:
        """
        Parses one or more documents into LlamaIndex Document objects.
        
        Args:
            file_paths (Union[str, List[str]]): Path or list of paths to process.
            
        Returns:
            List[Document]: The parsed LlamaIndex documents.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Filtering for existing files
        valid_paths = [path for path in file_paths if os.path.exists(path)]
        if not valid_paths:
            return []
        
        all_documents = []
        for path in valid_paths:
            # Parse satu file saja per panggilan agar kita yakin metadatanya benar
            docs = await self.parser.aload_data(path)
            for doc in docs:
                doc.metadata["file_name"] = os.path.basename(path)
                all_documents.append(doc)
                
        return all_documents

    def save_temp_file(self, file_content: bytes, filename: str) -> str:
        """
        Saves uploaded file bytes into a temporary data directory.
        
        Args:
            file_content (bytes): The raw file bytes.
            filename (str): Original filename.
            
        Returns:
            str: Absolute path to the saved file.
        """
        upload_dir = os.path.join(os.getcwd(), "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        return file_path
