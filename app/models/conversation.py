"""
Database model for conversations.
"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, Text
from sqlalchemy.orm import relationship
from app.models.base import Base
import uuid


class Conversation(Base):
    """
    Conversation model representing a user's workflow creation session.
    """
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    summary = Column(Text, nullable=True, doc="Summary of old messages")
    last_summarized_at = Column(DateTime, nullable=True, doc="When summary was last generated")
    user_id = Column(String(36), nullable=True, index=True, doc="For future multi-user support")
    is_deleted = Column(Boolean, default=False, nullable=False, doc="Soft delete flag")
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    workflow_state = relationship("WorkflowState", back_populates="conversation", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, created_at={self.created_at})>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "summary": self.summary,
            "last_summarized_at": self.last_summarized_at.isoformat() if self.last_summarized_at else None,
            "user_id": self.user_id,
            "is_deleted": self.is_deleted
        }