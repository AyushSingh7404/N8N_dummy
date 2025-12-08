"""
Database model for workflow states.
"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.sqlite import JSON
from app.models.base import Base
import uuid


class WorkflowState(Base):
    """
    WorkflowState model representing the current workflow JSON for a conversation.
    """
    __tablename__ = "workflow_states"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), unique=True, nullable=False, index=True)
    workflow_json = Column(JSON, nullable=False, doc="Current workflow structure")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    version = Column(Integer, default=1, nullable=False, doc="Version number for tracking edits")
    
    # Relationships
    conversation = relationship("Conversation", back_populates="workflow_state")
    
    def __repr__(self):
        return f"<WorkflowState(id={self.id}, conversation_id={self.conversation_id}, version={self.version})>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "workflow_json": self.workflow_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version
        }