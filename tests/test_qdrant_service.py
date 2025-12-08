"""
Tests for QdrantService.
"""

import pytest
from app.services.qdrant_service import QdrantService, QdrantException


class TestQdrantService:
    """Tests for Qdrant vector search."""
    
    def test_search_tools_success(self, mocker, sample_query_embedding):
        """Test successful tool search."""
        # Mock Qdrant client
        mock_client = mocker.MagicMock()
        mock_result = mocker.MagicMock()
        mock_result.id = "gmail_send-email"
        mock_result.score = 0.85
        mock_result.payload = {
            "tool_name": "gmail",
            "tool_slug": "gmail",
            "tool_display_name": "Gmail",
            "operation_name": "send_email",
            "operation_slug": "send-email",
            "operation_display_name": "Send Email",
            "category": "email",
            "operation_type": "action",
            "content": "Send an email",
            "required_fields": ["to", "subject"],
            "tags": ["email"],
            "auth_required": True
        }
        
        mock_client.search.return_value = [mock_result]
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        
        service = QdrantService()
        results = service.search_tools(sample_query_embedding)
        
        assert len(results) == 1
        assert results[0]["tool_name"] == "gmail"
        assert results[0]["score"] == 0.85
        mock_client.search.assert_called_once()
    
    def test_search_tools_with_metadata_filter(self, mocker, sample_query_embedding):
        """Test search with metadata filters."""
        mock_client = mocker.MagicMock()
        mock_client.search.return_value = []
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        
        service = QdrantService()
        results = service.search_tools(
            sample_query_embedding,
            metadata_filter={"category": "email"}
        )
        
        # Verify filter was passed
        call_args = mock_client.search.call_args
        assert call_args.kwargs['query_filter'] is not None
    
    def test_search_tools_invalid_embedding_dimension(self, mocker):
        """Test search with wrong embedding dimension."""
        mocker.patch('qdrant_client.QdrantClient')
        
        service = QdrantService()
        invalid_embedding = [0.1] * 512  # Wrong dimension
        
        with pytest.raises(ValueError, match="Invalid embedding dimension"):
            service.search_tools(invalid_embedding)
    
    def test_search_tools_retry_on_failure(self, mocker, sample_query_embedding):
        """Test retry logic on Qdrant failure."""
        mock_client = mocker.MagicMock()
        
        # Fail once, succeed on second attempt
        mock_result = mocker.MagicMock()
        mock_result.id = "test"
        mock_result.score = 0.5
        mock_result.payload = {
            "tool_name": "test",
            "tool_slug": "test",
            "tool_display_name": "Test",
            "operation_name": "test",
            "operation_slug": "test",
            "operation_display_name": "Test",
            "category": "test",
            "operation_type": "action",
            "content": "Test",
            "required_fields": [],
            "tags": [],
            "auth_required": False
        }
        
        mock_client.search.side_effect = [
            Exception("Connection error"),
            [mock_result]
        ]
        
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        mocker.patch('time.sleep')
        
        service = QdrantService()
        results = service.search_tools(sample_query_embedding)
        
        assert len(results) == 1
        assert mock_client.search.call_count == 2
    
    def test_search_tools_max_retries_exceeded(self, mocker, sample_query_embedding):
        """Test failure after max retries."""
        mock_client = mocker.MagicMock()
        mock_client.search.side_effect = Exception("Connection error")
        
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        mocker.patch('time.sleep')
        
        service = QdrantService()
        
        with pytest.raises(QdrantException, match="failed after 2 attempts"):
            service.search_tools(sample_query_embedding)
        
        assert mock_client.search.call_count == 2
    
    def test_filter_by_similarity_threshold_confident(self):
        """Test filtering with confident match."""
        service = QdrantService()
        
        results = [
            {"score": 0.85, "tool_display_name": "Gmail"},
            {"score": 0.65, "tool_display_name": "Slack"}
        ]
        
        filtered = service.filter_by_similarity_threshold(results)
        
        assert filtered["status"] == "confident"
        assert filtered["top_score"] == 0.85
    
    def test_filter_by_similarity_threshold_no_match(self):
        """Test filtering with no match."""
        service = QdrantService()
        
        results = [
            {"score": 0.3, "tool_display_name": "Gmail"}
        ]
        
        filtered = service.filter_by_similarity_threshold(results)
        
        assert filtered["status"] == "no_match"
        assert len(filtered["results"]) == 0
    
    def test_filter_by_similarity_threshold_ambiguous(self):
        """Test filtering with ambiguous results."""
        service = QdrantService()
        
        results = [
            {"score": 0.82, "tool_display_name": "Slack", "tool_slug": "slack"},
            {"score": 0.81, "tool_display_name": "Discord", "tool_slug": "discord"},
            {"score": 0.79, "tool_display_name": "Gmail", "tool_slug": "gmail"}
        ]
        
        filtered = service.filter_by_similarity_threshold(results)
        
        assert filtered["status"] == "ambiguous"
        assert len(filtered["results"]) == 3
        assert "suggestions" in filtered
    
    def test_filter_by_similarity_threshold_empty_results(self):
        """Test filtering with empty results."""
        service = QdrantService()
        
        filtered = service.filter_by_similarity_threshold([])
        
        assert filtered["status"] == "no_match"
        assert "message" in filtered
    
    def test_get_collection_info_success(self, mocker):
        """Test getting collection info."""
        mock_client = mocker.MagicMock()
        mock_collection = mocker.MagicMock()
        mock_collection.points_count = 100
        mock_collection.status = "green"
        mock_client.get_collection.return_value = mock_collection
        
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        
        service = QdrantService()
        info = service.get_collection_info()
        
        assert info["total_operations"] == 100
        assert info["status"] == "green"
    
    def test_get_collection_info_failure(self, mocker):
        """Test getting collection info when it fails."""
        mock_client = mocker.MagicMock()
        mock_client.get_collection.side_effect = Exception("Connection error")
        
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        
        service = QdrantService()
        
        with pytest.raises(QdrantException):
            service.get_collection_info()
    
    def test_health_check_success(self, mocker):
        """Test health check when Qdrant is healthy."""
        mock_client = mocker.MagicMock()
        mock_client.get_collection.return_value = mocker.MagicMock()
        
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        
        service = QdrantService()
        assert service.health_check() is True
    
    def test_health_check_failure(self, mocker):
        """Test health check when Qdrant is unavailable."""
        mock_client = mocker.MagicMock()
        mock_client.get_collection.side_effect = Exception("Connection error")
        
        mocker.patch('qdrant_client.QdrantClient', return_value=mock_client)
        
        service = QdrantService()
        assert service.health_check() is False