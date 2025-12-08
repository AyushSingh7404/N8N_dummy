"""
Tests for EmbeddingService.
"""

import pytest
from app.services.embedding_service import EmbeddingService, VoyageAIException


class TestEmbeddingService:
    """Tests for embedding generation."""
    
    def test_generate_embedding_success(self, mocker):
        """Test successful embedding generation."""
        # Mock Voyage AI client
        mock_client = mocker.MagicMock()
        mock_result = mocker.MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        
        mocker.patch('voyageai.Client', return_value=mock_client)
        
        service = EmbeddingService()
        embedding = service.generate_embedding("test query")
        
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)
        mock_client.embed.assert_called_once()
    
    def test_generate_embedding_empty_input(self):
        """Test embedding generation with empty input."""
        service = EmbeddingService()
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.generate_embedding("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.generate_embedding("   ")
    
    def test_generate_embedding_too_long(self):
        """Test embedding generation with text too long."""
        service = EmbeddingService()
        long_text = "a" * 9000
        
        with pytest.raises(ValueError, match="Text too long"):
            service.generate_embedding(long_text)
    
    def test_generate_embedding_retry_on_failure(self, mocker):
        """Test retry logic on temporary failure."""
        mock_client = mocker.MagicMock()
        
        # Fail twice, succeed on third attempt
        mock_result = mocker.MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        
        mock_client.embed.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            mock_result
        ]
        
        mocker.patch('voyageai.Client', return_value=mock_client)
        mocker.patch('time.sleep')  # Speed up test
        
        service = EmbeddingService()
        embedding = service.generate_embedding("test query")
        
        assert len(embedding) == 1024
        assert mock_client.embed.call_count == 3
    
    def test_generate_embedding_max_retries_exceeded(self, mocker):
        """Test failure after max retries."""
        mock_client = mocker.MagicMock()
        mock_client.embed.side_effect = Exception("Network error")
        
        mocker.patch('voyageai.Client', return_value=mock_client)
        mocker.patch('time.sleep')
        
        service = EmbeddingService()
        
        with pytest.raises(VoyageAIException, match="failed after 3 attempts"):
            service.generate_embedding("test query")
        
        assert mock_client.embed.call_count == 3
    
    def test_generate_embedding_invalid_dimension(self, mocker):
        """Test handling of invalid embedding dimension."""
        mock_client = mocker.MagicMock()
        mock_result = mocker.MagicMock()
        mock_result.embeddings = [[0.1] * 512]  # Wrong dimension
        mock_client.embed.return_value = mock_result
        
        mocker.patch('voyageai.Client', return_value=mock_client)
        
        service = EmbeddingService()
        
        with pytest.raises(VoyageAIException, match="Invalid embedding dimension"):
            service.generate_embedding("test query")
    
    def test_generate_batch_embeddings_success(self, mocker):
        """Test batch embedding generation."""
        mock_client = mocker.MagicMock()
        mock_result = mocker.MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_client.embed.return_value = mock_result
        
        mocker.patch('voyageai.Client', return_value=mock_client)
        
        service = EmbeddingService()
        texts = ["query 1", "query 2", "query 3"]
        embeddings = service.generate_batch_embeddings(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 1024 for emb in embeddings)
    
    def test_generate_batch_embeddings_empty_list(self):
        """Test batch generation with empty list."""
        service = EmbeddingService()
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            service.generate_batch_embeddings([])
    
    def test_generate_batch_embeddings_contains_empty(self):
        """Test batch generation with empty text in list."""
        service = EmbeddingService()
        
        with pytest.raises(ValueError, match="Text at index 1 is empty"):
            service.generate_batch_embeddings(["valid", "", "also valid"])
    
    def test_input_type_parameter(self, mocker):
        """Test that input_type parameter is passed correctly."""
        mock_client = mocker.MagicMock()
        mock_result = mocker.MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        
        mocker.patch('voyageai.Client', return_value=mock_client)
        
        service = EmbeddingService()
        
        # Test query type
        service.generate_embedding("test", input_type="query")
        call_args = mock_client.embed.call_args
        assert call_args.kwargs['input_type'] == "query"
        
        # Test document type
        service.generate_embedding("test", input_type="document")
        call_args = mock_client.embed.call_args
        assert call_args.kwargs['input_type'] == "document"