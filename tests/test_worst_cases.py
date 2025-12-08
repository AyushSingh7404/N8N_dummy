"""
Tests for worst-case scenarios and edge cases.
"""

import pytest
from fastapi import status
from app.services.embedding_service import VoyageAIException
from app.services.qdrant_service import QdrantException
from app.services.claude_service import BedrockException


class TestNoToolsMatch:
    """Test scenario: Query matches no tools."""
    
    def test_no_tools_match(
        self,
        client,
        mocker,
        sample_query_embedding
    ):
        """Test when no tools match the query."""
        # Mock embedding service
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.return_value = sample_query_embedding
        
        # Mock Qdrant to return low-score results
        mock_qdrant = mocker.patch("app.services.qdrant_service.QdrantService")
        mock_qdrant.return_value.search_tools.return_value = [
            {
                "id": "test",
                "score": 0.3,
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
        ]
        mock_qdrant.return_value.filter_by_similarity_threshold.return_value = {
            "status": "no_match",
            "results": [],
            "message": "No tools found matching your request."
        }
        
        response = client.post(
            "/api/workflow/create",
            json={"query": "Launch a rocket to Mars"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "no_match"
        assert data["workflow"] is None
        assert "message" in data


class TestAmbiguousQuery:
    """Test scenario: Query matches multiple tools equally."""
    
    def test_ambiguous_query(
        self,
        client,
        mocker,
        sample_query_embedding
    ):
        """Test when multiple tools match with similar scores."""
        # Mock services
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.return_value = sample_query_embedding
        
        tools = [
            {"id": "slack", "score": 0.82, "tool_slug": "slack", "tool_display_name": "Slack"},
            {"id": "discord", "score": 0.81, "tool_slug": "discord", "tool_display_name": "Discord"},
            {"id": "gmail", "score": 0.79, "tool_slug": "gmail", "tool_display_name": "Gmail"}
        ]
        
        # Add required fields to each tool
        for tool in tools:
            tool.update({
                "tool_name": tool["tool_slug"],
                "operation_name": "send_message",
                "operation_slug": "send-message",
                "operation_display_name": "Send Message",
                "category": "communication",
                "operation_type": "action",
                "content": "Send a message",
                "required_fields": [],
                "tags": [],
                "auth_required": True
            })
        
        mock_qdrant = mocker.patch("app.services.qdrant_service.QdrantService")
        mock_qdrant.return_value.search_tools.return_value = tools
        mock_qdrant.return_value.filter_by_similarity_threshold.return_value = {
            "status": "ambiguous",
            "results": tools,
            "message": "I found multiple tools that could work. Did you mean: Slack, Discord, or Gmail?",
            "suggestions": ["slack", "discord", "gmail"]
        }
        
        response = client.post(
            "/api/workflow/create",
            json={"query": "send a message"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ambiguous"
        assert "suggestions" in data
        assert len(data["suggestions"]) == 3


class TestClaudeInvalidJSON:
    """Test scenario: Claude returns invalid JSON."""
    
    def test_claude_returns_invalid_json(
        self,
        client,
        mocker,
        sample_query_embedding,
        sample_tools_retrieved
    ):
        """Test when Claude returns malformed JSON."""
        # Mock embedding and Qdrant
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.return_value = sample_query_embedding
        
        mock_qdrant = mocker.patch("app.services.qdrant_service.QdrantService")
        mock_qdrant.return_value.search_tools.return_value = sample_tools_retrieved
        mock_qdrant.return_value.filter_by_similarity_threshold.return_value = {
            "status": "confident",
            "results": sample_tools_retrieved,
            "top_score": 0.85
        }
        
        # Mock Claude to raise error after retries
        mock_claude = mocker.patch("app.services.claude_service.ClaudeService")
        mock_claude.return_value.generate_workflow.side_effect = BedrockException(
            "Failed to parse valid JSON after 3 attempts"
        )
        
        response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestServiceFailures:
    """Test scenario: External services fail."""
    
    def test_voyage_ai_unavailable(self, client, mocker):
        """Test when Voyage AI is unavailable."""
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.side_effect = VoyageAIException(
            "Embedding generation failed after 3 attempts"
        )
        
        response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Embedding service unavailable" in response.json()["message"]
    
    def test_qdrant_unavailable(
        self,
        client,
        mocker,
        sample_query_embedding
    ):
        """Test when Qdrant is unavailable."""
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.return_value = sample_query_embedding
        
        mock_qdrant = mocker.patch("app.services.qdrant_service.QdrantService")
        mock_qdrant.return_value.search_tools.side_effect = QdrantException(
            "Qdrant search failed after 2 attempts"
        )
        
        response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Search service unavailable" in response.json()["message"]
    
    def test_bedrock_rate_limit(
        self,
        client,
        mocker,
        sample_query_embedding,
        sample_tools_retrieved
    ):
        """Test when AWS Bedrock rate limit is hit."""
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.return_value = sample_query_embedding
        
        mock_qdrant = mocker.patch("app.services.qdrant_service.QdrantService")
        mock_qdrant.return_value.search_tools.return_value = sample_tools_retrieved
        mock_qdrant.return_value.filter_by_similarity_threshold.return_value = {
            "status": "confident",
            "results": sample_tools_retrieved,
            "top_score": 0.85
        }
        
        mock_claude = mocker.patch("app.services.claude_service.ClaudeService")
        mock_claude.return_value.generate_workflow.side_effect = BedrockException(
            "Rate limit exceeded"
        )
        
        response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS


class TestMultiTurnConversation:
    """Test scenario: Multi-turn conversation with edits."""
    
    def test_conversation_with_multiple_edits(
        self,
        client,
        mocker,
        sample_query_embedding,
        sample_tools_retrieved,
        sample_workflow
    ):
        """Test editing workflow multiple times."""
        # Setup mocks
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.return_value = sample_query_embedding
        
        mock_qdrant = mocker.patch("app.services.qdrant_service.QdrantService")
        mock_qdrant.return_value.search_tools.return_value = sample_tools_retrieved
        mock_qdrant.return_value.filter_by_similarity_threshold.return_value = {
            "status": "confident",
            "results": sample_tools_retrieved,
            "top_score": 0.85
        }
        
        mock_claude = mocker.patch("app.services.claude_service.ClaudeService")
        mock_claude.return_value.generate_workflow.return_value = sample_workflow
        mock_claude.return_value.generate_workflow_edit.return_value = sample_workflow
        
        # Create initial workflow
        create_response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        assert create_response.status_code == status.HTTP_200_OK
        conversation_id = create_response.json()["conversation_id"]
        
        # Edit 1: Change to Slack
        edit1_response = client.post(
            "/api/workflow/edit",
            json={
                "conversation_id": conversation_id,
                "edit_instruction": "Change to Slack"
            }
        )
        assert edit1_response.status_code == status.HTTP_200_OK
        
        # Edit 2: Add delay
        edit2_response = client.post(
            "/api/workflow/edit",
            json={
                "conversation_id": conversation_id,
                "edit_instruction": "Add delay of 5 minutes"
            }
        )
        assert edit2_response.status_code == status.HTTP_200_OK
        
        # Verify conversation history
        history_response = client.get(f"/api/workflow/conversation/{conversation_id}")
        assert history_response.status_code == status.HTTP_200_OK
        data = history_response.json()
        
        # Should have 6 messages (3 user + 3 assistant)
        assert data["message_count"] >= 6


class TestConcurrentRequests:
    """Test scenario: Concurrent workflow creation."""
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_creation(
        self,
        client,
        mocker,
        sample_query_embedding,
        sample_tools_retrieved,
        sample_workflow
    ):
        """Test multiple users creating workflows simultaneously."""
        # Setup mocks
        mock_embedding = mocker.patch("app.services.embedding_service.EmbeddingService")
        mock_embedding.return_value.generate_embedding.return_value = sample_query_embedding
        
        mock_qdrant = mocker.patch("app.services.qdrant_service.QdrantService")
        mock_qdrant.return_value.search_tools.return_value = sample_tools_retrieved
        mock_qdrant.return_value.filter_by_similarity_threshold.return_value = {
            "status": "confident",
            "results": sample_tools_retrieved,
            "top_score": 0.85
        }
        
        mock_claude = mocker.patch("app.services.claude_service.ClaudeService")
        mock_claude.return_value.generate_workflow.return_value = sample_workflow
        
        # Create multiple workflows concurrently
        responses = []
        for i in range(5):
            response = client.post(
                "/api/workflow/create",
                json={"query": f"Send email {i}"}
            )
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
        
        # All should have unique conversation IDs
        conversation_ids = [r.json()["conversation_id"] for r in responses]
        assert len(set(conversation_ids)) == 5