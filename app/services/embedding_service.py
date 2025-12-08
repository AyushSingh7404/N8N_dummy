"""
Embedding service for generating text embeddings using Voyage AI.
"""

import time
from typing import List
import voyageai
from config import get_settings


class VoyageAIException(Exception):
    """Custom exception for Voyage AI errors."""
    pass


class EmbeddingService:
    """Service for generating embeddings via Voyage AI."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.settings = get_settings()
        self.client = voyageai.Client(api_key=self.settings.voyage_ai_key)
        self.model = self.settings.voyage_model
        self.max_retries = 3
        self.base_retry_delay = 1  # seconds
    
    def generate_embedding(
        self, 
        text: str, 
        input_type: str = "query"
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            input_type: Either "query" (for search queries) or "document" (for tool docs)
            
        Returns:
            List[float]: Embedding vector (1024 dimensions)
            
        Raises:
            ValueError: If text is empty or too long
            VoyageAIException: If embedding generation fails after retries
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > 8000:
            raise ValueError(f"Text too long: {len(text)} characters (max 8000)")
        
        # Generate embedding with retry logic
        for attempt in range(self.max_retries):
            try:
                result = self.client.embed(
                    texts=[text],
                    model=self.model,
                    input_type=input_type
                )
                
                # Validate response
                if not result.embeddings or len(result.embeddings) == 0:
                    raise VoyageAIException("Empty embedding response")
                
                embedding = result.embeddings[0]
                
                # Validate embedding dimension
                if len(embedding) != self.settings.embedding_dimension:
                    raise VoyageAIException(
                        f"Invalid embedding dimension: {len(embedding)} "
                        f"(expected {self.settings.embedding_dimension})"
                    )
                
                return embedding
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = self.base_retry_delay * (2 ** attempt)
                    print(f"Embedding error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    raise VoyageAIException(
                        f"Embedding generation failed after {self.max_retries} attempts: {str(e)}"
                    )
    
    def generate_batch_embeddings(
        self, 
        texts: List[str], 
        input_type: str = "query"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            input_type: Either "query" or "document"
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty or contains invalid texts
            VoyageAIException: If embedding generation fails
        """
        # Validate input
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
            if len(text) > 8000:
                raise ValueError(f"Text at index {i} is too long: {len(text)} characters")
        
        # Voyage AI supports up to 128 texts per batch
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate embeddings with retry logic
            for attempt in range(self.max_retries):
                try:
                    result = self.client.embed(
                        texts=batch,
                        model=self.model,
                        input_type=input_type
                    )
                    
                    if not result.embeddings or len(result.embeddings) != len(batch):
                        raise VoyageAIException(
                            f"Invalid batch response: got {len(result.embeddings)} embeddings, "
                            f"expected {len(batch)}"
                        )
                    
                    all_embeddings.extend(result.embeddings)
                    break
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait_time = self.base_retry_delay * (2 ** attempt)
                        print(f"Batch embedding error (attempt {attempt + 1}/{self.max_retries}): {e}")
                        time.sleep(wait_time)
                    else:
                        raise VoyageAIException(
                            f"Batch embedding failed after {self.max_retries} attempts: {str(e)}"
                        )
        
        return all_embeddings
    
    async def generate_embedding_async(
        self, 
        text: str, 
        input_type: str = "query"
    ) -> List[float]:
        """
        Async version of generate_embedding for use in FastAPI.
        
        Note: Voyage AI client is synchronous, so this wraps the sync call.
        For true async, you'd need an async HTTP client.
        
        Args:
            text: Text to embed
            input_type: "query" or "document"
            
        Returns:
            List[float]: Embedding vector
        """
        # For now, just call synchronous version
        # In production, consider using asyncio.to_thread() or an async HTTP client
        return self.generate_embedding(text, input_type)