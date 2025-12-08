"""
Health check and utility API routes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import json
from pathlib import Path

from app.models.base import get_db_dependency
from app.schemas.response_schemas import HealthResponse, ToolsListResponse, ToolInfo
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
from config import get_settings
from app.utils.logger import get_logger

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db_dependency)):
    """
    Check health of all system components.
    
    Returns:
        HealthResponse with status of each service
    """
    services_status = {}
    overall_healthy = True
    
    # Check Qdrant
    try:
        qdrant_service = QdrantService()
        if qdrant_service.health_check():
            services_status["qdrant"] = "healthy"
            logger.debug("Qdrant health check: OK")
        else:
            services_status["qdrant"] = "unavailable"
            overall_healthy = False
            logger.warning("Qdrant health check: Failed")
    except Exception as e:
        services_status["qdrant"] = "unavailable"
        overall_healthy = False
        logger.error(f"Qdrant health check error: {e}")
    
    # Check Database
    try:
        db.execute(text("SELECT 1"))
        services_status["database"] = "healthy"
        logger.debug("Database health check: OK")
    except Exception as e:
        services_status["database"] = "unavailable"
        overall_healthy = False
        logger.error(f"Database health check error: {e}")
    
    # Check Embedding Service (optional - can be slow)
    # Uncomment if you want to test Voyage AI on every health check
    # try:
    #     embedding_service = EmbeddingService()
    #     test_embedding = embedding_service.generate_embedding("test")
    #     if len(test_embedding) == 1024:
    #         services_status["embedding"] = "healthy"
    #     else:
    #         services_status["embedding"] = "degraded"
    #         overall_healthy = False
    # except Exception as e:
    #     services_status["embedding"] = "unavailable"
    #     overall_healthy = False
    #     logger.error(f"Embedding service health check error: {e}")
    
    # For now, assume embedding service is healthy if configured
    settings = get_settings()
    if settings.voyage_ai_key and settings.voyage_ai_key != "your_voyage_key_here":
        services_status["embedding"] = "configured"
    else:
        services_status["embedding"] = "not_configured"
        overall_healthy = False
    
    # Determine overall status
    if overall_healthy:
        overall_status = "healthy"
    elif all(s == "unavailable" for s in services_status.values()):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        services=services_status,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/api/tools", response_model=ToolsListResponse)
async def get_tools_list():
    """
    Get list of all available tools with their operations.
    
    Returns:
        ToolsListResponse with all tools
    """
    try:
        settings = get_settings()
        tools_path = Path(settings.tools_json_path)
        
        if not tools_path.exists():
            logger.error(f"Tools JSON not found at: {tools_path}")
            return ToolsListResponse(tools=[], total_count=0)
        
        # Load tools JSON
        with open(tools_path, 'r') as f:
            tools_data = json.load(f)
        
        # Format tools for response
        tools_list = []
        for tool in tools_data:
            tool_info = ToolInfo(
                name=tool.get("name", ""),
                slug=tool.get("slug", ""),
                displayName=tool.get("displayName", ""),
                description=tool.get("description", ""),
                category=tool.get("category", "general"),
                icon_url=tool.get("iconUrl", ""),
                operations=[
                    {
                        "name": op.get("name"),
                        "slug": op.get("slug"),
                        "displayName": op.get("displayName"),
                        "description": op.get("description"),
                        "operationType": op.get("operationType")
                    }
                    for op in tool.get("operations", [])
                ],
                auth_required=tool.get("authConfig", {}).get("type") != "none"
            )
            tools_list.append(tool_info)
        
        logger.info(f"Returned {len(tools_list)} tools")
        
        return ToolsListResponse(
            tools=tools_list,
            total_count=len(tools_list)
        )
        
    except Exception as e:
        logger.error(f"Error loading tools list: {e}", exc_info=True)
        return ToolsListResponse(tools=[], total_count=0)


@router.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Dict with API info
    """
    return {
        "name": "RAG Tool Retrieval API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "create_workflow": "/api/workflow/create",
            "edit_workflow": "/api/workflow/edit",
            "get_conversation": "/api/workflow/conversation/{conversation_id}",
            "delete_conversation": "/api/workflow/conversation/{conversation_id}",
            "tools_list": "/api/tools",
            "docs": "/docs"
        }
    }