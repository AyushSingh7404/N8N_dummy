"""
Conversation service for managing conversation history and workflow state.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid

from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.models.workflow import WorkflowState


class ConversationService:
    """Service for managing conversations and workflow state."""
    
    def __init__(self, db: Session):
        """
        Initialize conversation service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def create_conversation(self, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            user_id: Optional user ID for multi-user support
            
        Returns:
            str: Conversation ID (UUID)
        """
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id
        )
        
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        
        return conversation.id
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get conversation by ID.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            Conversation object or None if not found
        """
        return self.db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.is_deleted == False
        ).first()
    
    def get_conversation_history(
        self, 
        conversation_id: str, 
        last_n: int = 5
    ) -> Dict[str, Any]:
        """
        Get conversation history with last N messages.
        
        Args:
            conversation_id: Conversation UUID
            last_n: Number of recent messages to retrieve
            
        Returns:
            Dict with:
                - conversation: Conversation object
                - messages: List of last N messages
                - summary: Summary of older messages (if exists)
                - has_more: Boolean indicating if there are older messages
                - total_messages: Total message count
        """
        # Get conversation
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        
        # Get total message count
        total_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).count()
        
        # Get last N messages (ordered by timestamp)
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(desc(Message.timestamp)).limit(last_n).all()
        
        # Reverse to get chronological order (oldest first)
        messages = list(reversed(messages))
        
        return {
            "conversation": conversation,
            "messages": [msg.to_dict() for msg in messages],
            "summary": conversation.summary,
            "has_more": total_messages > last_n,
            "total_messages": total_messages
        }
    
    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tools_retrieved: Optional[List[str]] = None,
        similarity_scores: Optional[Dict[str, float]] = None
    ) -> Message:
        """
        Save a message to the conversation.
        
        Args:
            conversation_id: Conversation UUID
            role: "user" or "assistant"
            content: Message content
            tools_retrieved: Optional list of tool IDs retrieved
            similarity_scores: Optional dict of similarity scores
            
        Returns:
            Message: Created message object
        """
        # Validate conversation exists
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Create message
        message = Message(
            conversation_id=conversation_id,
            role=MessageRole(role),
            content=content,
            tools_retrieved=tools_retrieved,
            similarity_scores=similarity_scores
        )
        
        self.db.add(message)
        
        # Update conversation updated_at
        conversation.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(message)
        
        # Check if summarization needed (background task trigger)
        self._check_summarization_needed(conversation_id)
        
        return message
    
    def save_workflow(
        self,
        conversation_id: str,
        workflow_json: Dict[str, Any]
    ) -> WorkflowState:
        """
        Save or update workflow state for a conversation.
        
        Args:
            conversation_id: Conversation UUID
            workflow_json: Workflow JSON structure
            
        Returns:
            WorkflowState: Created or updated workflow state
        """
        # Check if workflow already exists
        workflow_state = self.db.query(WorkflowState).filter(
            WorkflowState.conversation_id == conversation_id
        ).first()
        
        if workflow_state:
            # Update existing workflow
            workflow_state.workflow_json = workflow_json
            workflow_state.version += 1
            workflow_state.updated_at = datetime.utcnow()
        else:
            # Create new workflow
            workflow_state = WorkflowState(
                conversation_id=conversation_id,
                workflow_json=workflow_json
            )
            self.db.add(workflow_state)
        
        self.db.commit()
        self.db.refresh(workflow_state)
        
        return workflow_state
    
    def get_current_workflow(
        self,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current workflow for a conversation.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            Dict: Workflow JSON or None if not found
        """
        workflow_state = self.db.query(WorkflowState).filter(
            WorkflowState.conversation_id == conversation_id
        ).first()
        
        return workflow_state.workflow_json if workflow_state else None
    
    def update_summary(
        self,
        conversation_id: str,
        summary: str
    ) -> None:
        """
        Update conversation summary.
        
        Args:
            conversation_id: Conversation UUID
            summary: Summary text
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        conversation.summary = summary
        conversation.last_summarized_at = datetime.utcnow()
        
        self.db.commit()
    
    def _check_summarization_needed(self, conversation_id: str) -> bool:
        """
        Check if conversation needs summarization.
        
        This is a trigger point for background summarization tasks.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            bool: True if summarization needed
        """
        # Get total messages
        total_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).count()
        
        # Get conversation
        conversation = self.get_conversation(conversation_id)
        
        # Trigger summarization if:
        # 1. More than 10 messages exist
        # 2. Either no summary exists OR last summarization was >5 messages ago
        if total_messages > 10:
            if not conversation.last_summarized_at:
                # Never summarized
                return True
            
            # Count messages since last summarization
            messages_since = self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.timestamp > conversation.last_summarized_at
            ).count()
            
            if messages_since >= 5:
                return True
        
        return False
    
    def get_messages_for_summarization(
        self,
        conversation_id: str,
        exclude_last_n: int = 5
    ) -> List[Dict[str, str]]:
        """
        Get messages that should be summarized (all except last N).
        
        Args:
            conversation_id: Conversation UUID
            exclude_last_n: Number of recent messages to exclude
            
        Returns:
            List of message dicts with role and content
        """
        # Get total count
        total_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).count()
        
        # Calculate how many to retrieve
        messages_to_summarize = max(0, total_messages - exclude_last_n)
        
        if messages_to_summarize == 0:
            return []
        
        # Get messages (excluding last N)
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).limit(messages_to_summarize).all()
        
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Soft delete a conversation.
        
        Args:
            conversation_id: Conversation UUID
            
        Returns:
            bool: True if deleted, False if not found
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation.is_deleted = True
        conversation.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True