"""
Tests for ConversationService.
"""

import pytest
from app.services.conversation_service import ConversationService
from app.models.message import MessageRole


class TestConversationService:
    """Tests for conversation management."""
    
    def test_create_conversation(self, test_db):
        """Test creating a new conversation."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        assert conversation_id is not None
        assert len(conversation_id) == 36  # UUID length
    
    def test_get_conversation_success(self, test_db):
        """Test getting an existing conversation."""
        service = ConversationService(test_db)
        
        # Create conversation
        conversation_id = service.create_conversation()
        
        # Get conversation
        conversation = service.get_conversation(conversation_id)
        
        assert conversation is not None
        assert conversation.id == conversation_id
        assert conversation.is_deleted is False
    
    def test_get_conversation_not_found(self, test_db):
        """Test getting non-existent conversation."""
        service = ConversationService(test_db)
        
        conversation = service.get_conversation("550e8400-e29b-41d4-a716-446655440000")
        
        assert conversation is None
    
    def test_save_message(self, test_db):
        """Test saving a message."""
        service = ConversationService(test_db)
        
        # Create conversation
        conversation_id = service.create_conversation()
        
        # Save message
        message = service.save_message(
            conversation_id=conversation_id,
            role="user",
            content="Send email",
            tools_retrieved=["gmail"],
            similarity_scores={"gmail": 0.85}
        )
        
        assert message is not None
        assert message.role == MessageRole.USER
        assert message.content == "Send email"
        assert message.tools_retrieved == ["gmail"]
    
    def test_save_message_invalid_conversation(self, test_db):
        """Test saving message to non-existent conversation."""
        service = ConversationService(test_db)
        
        with pytest.raises(ValueError, match="Conversation not found"):
            service.save_message(
                conversation_id="550e8400-e29b-41d4-a716-446655440000",
                role="user",
                content="Test"
            )
    
    def test_get_conversation_history(self, test_db):
        """Test getting conversation history."""
        service = ConversationService(test_db)
        
        # Create conversation and messages
        conversation_id = service.create_conversation()
        
        for i in range(10):
            service.save_message(
                conversation_id=conversation_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )
        
        # Get history (last 5)
        history = service.get_conversation_history(conversation_id, last_n=5)
        
        assert history is not None
        assert len(history["messages"]) == 5
        assert history["total_messages"] == 10
        assert history["has_more"] is True
    
    def test_get_conversation_history_chronological_order(self, test_db):
        """Test that history is in chronological order."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        # Save messages
        for i in range(5):
            service.save_message(
                conversation_id=conversation_id,
                role="user",
                content=f"Message {i}"
            )
        
        history = service.get_conversation_history(conversation_id)
        
        # Should be oldest first
        messages = history["messages"]
        assert messages[0]["content"] == "Message 0"
        assert messages[-1]["content"] == "Message 4"
    
    def test_save_workflow(self, test_db, sample_workflow):
        """Test saving a workflow."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        workflow_state = service.save_workflow(
            conversation_id=conversation_id,
            workflow_json=sample_workflow
        )
        
        assert workflow_state is not None
        assert workflow_state.workflow_json == sample_workflow
        assert workflow_state.version == 1
    
    def test_update_workflow(self, test_db, sample_workflow):
        """Test updating an existing workflow."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        # Save initial workflow
        service.save_workflow(conversation_id, sample_workflow)
        
        # Update workflow
        updated_workflow = {
            "nodes": [{"id": "node2", "type": "slack.send-message"}],
            "connections": {}
        }
        
        workflow_state = service.save_workflow(
            conversation_id=conversation_id,
            workflow_json=updated_workflow
        )
        
        assert workflow_state.workflow_json == updated_workflow
        assert workflow_state.version == 2  # Version incremented
    
    def test_get_current_workflow(self, test_db, sample_workflow):
        """Test getting current workflow."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        service.save_workflow(conversation_id, sample_workflow)
        
        workflow = service.get_current_workflow(conversation_id)
        
        assert workflow == sample_workflow
    
    def test_get_current_workflow_not_found(self, test_db):
        """Test getting workflow when none exists."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        workflow = service.get_current_workflow(conversation_id)
        
        assert workflow is None
    
    def test_update_summary(self, test_db):
        """Test updating conversation summary."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        service.update_summary(
            conversation_id=conversation_id,
            summary="User wants to send emails"
        )
        
        conversation = service.get_conversation(conversation_id)
        
        assert conversation.summary == "User wants to send emails"
        assert conversation.last_summarized_at is not None
    
    def test_get_messages_for_summarization(self, test_db):
        """Test getting messages for summarization."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        # Create 10 messages
        for i in range(10):
            service.save_message(
                conversation_id=conversation_id,
                role="user",
                content=f"Message {i}"
            )
        
        # Get messages for summarization (exclude last 5)
        messages = service.get_messages_for_summarization(
            conversation_id,
            exclude_last_n=5
        )
        
        assert len(messages) == 5
        assert messages[0]["content"] == "Message 0"
        assert messages[-1]["content"] == "Message 4"
    
    def test_delete_conversation(self, test_db):
        """Test soft deleting a conversation."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        # Delete conversation
        success = service.delete_conversation(conversation_id)
        
        assert success is True
        
        # Verify it's marked as deleted
        conversation = service.get_conversation(conversation_id)
        assert conversation is None  # get_conversation filters deleted
    
    def test_delete_nonexistent_conversation(self, test_db):
        """Test deleting non-existent conversation."""
        service = ConversationService(test_db)
        
        success = service.delete_conversation("550e8400-e29b-41d4-a716-446655440000")
        
        assert success is False
    
    def test_check_summarization_needed(self, test_db):
        """Test summarization trigger logic."""
        service = ConversationService(test_db)
        
        conversation_id = service.create_conversation()
        
        # Add 5 messages - should not trigger
        for i in range(5):
            service.save_message(
                conversation_id=conversation_id,
                role="user",
                content=f"Message {i}"
            )
        
        # Add 6 more (total 11) - should trigger
        for i in range(5, 11):
            service.save_message(
                conversation_id=conversation_id,
                role="user",
                content=f"Message {i}"
            )
        
        # Verify summarization is needed
        needs_summary = service._check_summarization_needed(conversation_id)
        assert needs_summary is True
    
    def test_multiple_conversations_isolated(self, test_db):
        """Test that multiple conversations are isolated."""
        service = ConversationService(test_db)
        
        # Create two conversations
        conv1_id = service.create_conversation()
        conv2_id = service.create_conversation()
        
        # Add messages to each
        service.save_message(conv1_id, "user", "Conv 1 message")
        service.save_message(conv2_id, "user", "Conv 2 message")
        
        # Get histories
        history1 = service.get_conversation_history(conv1_id)
        history2 = service.get_conversation_history(conv2_id)
        
        # Verify isolation
        assert len(history1["messages"]) == 1
        assert len(history2["messages"]) == 1
        assert history1["messages"][0]["content"] != history2["messages"][0]["content"]