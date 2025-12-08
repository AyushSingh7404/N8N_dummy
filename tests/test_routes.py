"""
Tests for API routes.
"""

import pytest
from fastapi import status


class TestWorkflowCreation:
    """Tests for workflow creation endpoint."""
    
    def test_create_workflow_success(
        self, 
        client, 
        mock_embedding_service,
        mock_qdrant_service,
        mock_claude_service
    ):
        """Test successful workflow creation."""
        response = client.post(
            "/api/workflow/create",
            json={"query": "Send an email when form is submitted"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "conversation_id" in data
        assert data["status"] == "confident"
        assert data["workflow"] is not None
        assert len(data["tools_used"]) > 0
    
    def test_create_workflow_empty_query(self, client):
        """Test workflow creation with empty query."""
        response = client.post(
            "/api/workflow/create",
            json={"query": ""}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_create_workflow_invalid_conversation_id(self, client):
        """Test workflow creation with invalid conversation_id."""
        response = client.post(
            "/api/workflow/create",
            json={
                "query": "Send email",
                "conversation_id": "invalid-uuid"
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestWorkflowEdit:
    """Tests for workflow editing endpoint."""
    
    def test_edit_workflow_success(
        self,
        client,
        test_db,
        mock_embedding_service,
        mock_qdrant_service,
        mock_claude_service
    ):
        """Test successful workflow edit."""
        # First create a workflow
        create_response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        conversation_id = create_response.json()["conversation_id"]
        
        # Then edit it
        edit_response = client.post(
            "/api/workflow/edit",
            json={
                "conversation_id": conversation_id,
                "edit_instruction": "Change to Slack"
            }
        )
        
        assert edit_response.status_code == status.HTTP_200_OK
        data = edit_response.json()
        assert data["workflow"] is not None
    
    def test_edit_workflow_not_found(self, client):
        """Test editing non-existent workflow."""
        response = client.post(
            "/api/workflow/edit",
            json={
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "edit_instruction": "Change it"
            }
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConversation:
    """Tests for conversation endpoints."""
    
    def test_get_conversation_success(
        self,
        client,
        mock_embedding_service,
        mock_qdrant_service,
        mock_claude_service
    ):
        """Test getting conversation details."""
        # Create a workflow first
        create_response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        conversation_id = create_response.json()["conversation_id"]
        
        # Get conversation
        response = client.get(f"/api/workflow/conversation/{conversation_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["conversation_id"] == conversation_id
        assert len(data["messages"]) > 0
    
    def test_get_conversation_not_found(self, client):
        """Test getting non-existent conversation."""
        response = client.get(
            "/api/workflow/conversation/550e8400-e29b-41d4-a716-446655440000"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_conversation_success(
        self,
        client,
        mock_embedding_service,
        mock_qdrant_service,
        mock_claude_service
    ):
        """Test deleting a conversation."""
        # Create a workflow first
        create_response = client.post(
            "/api/workflow/create",
            json={"query": "Send email"}
        )
        conversation_id = create_response.json()["conversation_id"]
        
        # Delete it
        response = client.delete(f"/api/workflow/conversation/{conversation_id}")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True


class TestHealth:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_tools_list(self, client):
        """Test tools list endpoint."""
        response = client.get("/api/tools")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "tools" in data
        assert "total_count" in data