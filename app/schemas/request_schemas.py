"""
Request schemas for API validation using Pydantic.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
import re


class CreateWorkflowRequest(BaseModel):
    """Request schema for creating a new workflow."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's natural language request for workflow creation"
    )
    
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID to continue existing conversation"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        """Validate conversation_id is a valid UUID format."""
        if v is not None:
            # UUID v4 format validation
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
            if not re.match(uuid_pattern, v, re.IGNORECASE):
                raise ValueError("Invalid conversation_id format (must be UUID)")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Send an email when a form is submitted",
                "conversation_id": None
            }
        }


class EditWorkflowRequest(BaseModel):
    """Request schema for editing an existing workflow."""
    
    conversation_id: str = Field(
        ...,
        description="Conversation ID of the workflow to edit"
    )
    
    edit_instruction: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User's instruction for editing the workflow"
    )
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        """Validate conversation_id is a valid UUID format."""
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v, re.IGNORECASE):
            raise ValueError("Invalid conversation_id format (must be UUID)")
        return v
    
    @validator('edit_instruction')
    def validate_edit_instruction(cls, v):
        """Validate edit instruction is not empty."""
        if not v or not v.strip():
            raise ValueError("Edit instruction cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "edit_instruction": "Change Gmail to Slack"
            }
        }