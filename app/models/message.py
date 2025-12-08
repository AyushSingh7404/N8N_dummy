"""
Database model for messages.
"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.sqlite import JSON
from app.models.base import Base
import uuid
import enum


class MessageRole(str, enum.Enum):
    """Message role enum."""
    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    """
    Message model representing a single message in a conversation.
    """
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    tools_retrieved = Column(JSON, nullable=True, doc="Array of tool IDs retrieved from RAG")
    similarity_scores = Column(JSON, nullable=True, doc="Similarity scores from Qdrant")
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Composite index for efficient querying
    __table_args__ = (
        Index('ix_messages_conversation_timestamp', 'conversation_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, conversation_id={self.conversation_id})>"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tools_retrieved": self.tools_retrieved,
            "similarity_scores": self.similarity_scores
        }