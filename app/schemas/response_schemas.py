"""
Response schemas for API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class WorkflowResponse(BaseModel):
    """Response schema for workflow creation/editing."""
    
    conversation_id: str = Field(..., description="Conversation ID")
    workflow: Optional[Dict[str, Any]] = Field(None, description="Generated workflow JSON")
    tools_used: List[str] = Field(default_factory=list, description="List of tool slugs used")
    confidence_score: float = Field(..., description="Confidence score (0.0 to 1.0)")
    status: str = Field(..., description="Status: confident | ambiguous | no_match")
    message: Optional[str] = Field(None, description="Clarification message if needed")
    suggestions: Optional[List[str]] = Field(None, description="Tool suggestions if ambiguous")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "workflow": {
                    "nodes": [
                        {
                            "id": "node1",
                            "type": "gmail.send-email",
                            "displayName": "Send Email",
                            "parameters": {
                                "to": "user@example.com",
                                "subject": "Test",
                                "body": "Hello"
                            }
                        }
                    ],
                    "connections": {}
                },
                "tools_used": ["gmail"],
                "confidence_score": 0.87,
                "status": "confident",
                "message": None
            }
        }


class ConversationResponse(BaseModel):
    """Response schema for conversation details."""
    
    conversation_id: str
    messages: List[Dict[str, Any]]
    workflow: Optional[Dict[str, Any]]
    summary: Optional[str]
    created_at: str
    message_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "messages": [
                    {
                        "role": "user",
                        "content": "Send email when form submitted",
                        "timestamp": "2024-12-07T10:00:00Z"
                    }
                ],
                "workflow": {"nodes": [], "connections": {}},
                "summary": None,
                "created_at": "2024-12-07T10:00:00Z",
                "message_count": 2
            }
        }


class ToolInfo(BaseModel):
    """Schema for tool information."""
    
    name: str
    slug: str
    displayName: str
    description: str
    category: str
    icon_url: str
    operations: List[Dict[str, Any]]
    auth_required: bool


class ToolsListResponse(BaseModel):
    """Response schema for tools list."""
    
    tools: List[ToolInfo]
    total_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "tools": [
                    {
                        "name": "gmail",
                        "slug": "gmail",
                        "displayName": "Gmail",
                        "description": "Send and manage emails",
                        "category": "email",
                        "icon_url": "https://example.com/gmail.svg",
                        "operations": [],
                        "auth_required": True
                    }
                ],
                "total_count": 5
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Overall status: healthy | degraded | unhealthy")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    timestamp: str = Field(..., description="Timestamp of health check")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "qdrant": "healthy",
                    "database": "healthy",
                    "embedding": "healthy"
                },
                "timestamp": "2024-12-07T10:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid conversation_id format",
                "details": {
                    "field": "conversation_id",
                    "expected": "UUID format"
                }
            }
        }


class DeleteResponse(BaseModel):
    """Response schema for delete operations."""
    
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Result message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Conversation deleted successfully"
            }
        }