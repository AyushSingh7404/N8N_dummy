"""
Workflow API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
import time

from app.models.base import get_db_dependency
from app.schemas.request_schemas import CreateWorkflowRequest, EditWorkflowRequest
from app.schemas.response_schemas import WorkflowResponse, ConversationResponse, DeleteResponse
from app.services.embedding_service import EmbeddingService, VoyageAIException
from app.services.qdrant_service import QdrantService, QdrantException
from app.services.claude_service import ClaudeService, BedrockException
from app.services.conversation_service import ConversationService
from app.utils.logger import get_logger

router = APIRouter(prefix="/api/workflow", tags=["workflow"])
logger = get_logger(__name__)


@router.post("/create", response_model=WorkflowResponse)
async def create_workflow(
    request: CreateWorkflowRequest,
    db: Session = Depends(get_db_dependency)
):
    """
    Create a new workflow from natural language query.
    
    Args:
        request: CreateWorkflowRequest with query and optional conversation_id
        db: Database session
        
    Returns:
        WorkflowResponse with generated workflow
        
    Raises:
        HTTPException: Various error codes based on failure type
    """
    start_time = time.time()
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        qdrant_service = QdrantService()
        claude_service = ClaudeService()
        conversation_service = ConversationService(db)
        
        # Log request
        logger.info(f"Workflow creation request: '{request.query[:100]}...'")
        
        # Step 1: Get or create conversation
        if request.conversation_id:
            # Load existing conversation
            conversation = conversation_service.get_conversation(request.conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation not found: {request.conversation_id}"
                )
            
            # Load conversation history
            history_data = conversation_service.get_conversation_history(
                request.conversation_id,
                last_n=5
            )
            conversation_history = history_data["messages"]
            conversation_id = request.conversation_id
            
            logger.info(f"Continuing conversation: {conversation_id}")
        else:
            # Create new conversation
            conversation_id = conversation_service.create_conversation()
            conversation_history = []
            logger.info(f"Created new conversation: {conversation_id}")
        
        # Step 2: Generate embedding for query
        try:
            query_embedding = embedding_service.generate_embedding(
                request.query,
                input_type="query"
            )
            logger.debug(f"Generated query embedding: {len(query_embedding)} dimensions")
        except VoyageAIException as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Embedding service unavailable: {str(e)}"
            )
        
        # Step 3: Search tools in Qdrant
        try:
            search_results = qdrant_service.search_tools(query_embedding)
            logger.info(f"Retrieved {len(search_results)} tools from Qdrant")
            
            # Log top results
            if search_results:
                top_result = search_results[0]
                logger.debug(
                    f"Top result: {top_result['tool_display_name']} - "
                    f"{top_result['operation_display_name']} (score: {top_result['score']:.4f})"
                )
        except QdrantException as e:
            logger.error(f"Qdrant search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Search service unavailable: {str(e)}"
            )
        
        # Step 4: Filter by similarity threshold
        filtered_results = qdrant_service.filter_by_similarity_threshold(search_results)
        result_status = filtered_results["status"]
        
        logger.info(f"Result status: {result_status}")
        
        # Handle no match
        if result_status == "no_match":
            # Save user message
            conversation_service.save_message(
                conversation_id=conversation_id,
                role="user",
                content=request.query,
                tools_retrieved=[],
                similarity_scores={}
            )
            
            # Save assistant response
            conversation_service.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=filtered_results["message"]
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"No tools matched (completed in {elapsed_time:.2f}s)")
            
            return WorkflowResponse(
                conversation_id=conversation_id,
                workflow=None,
                tools_used=[],
                confidence_score=0.0,
                status="no_match",
                message=filtered_results["message"]
            )
        
        # Handle ambiguous results
        if result_status == "ambiguous":
            # Save user message
            conversation_service.save_message(
                conversation_id=conversation_id,
                role="user",
                content=request.query,
                tools_retrieved=filtered_results.get("suggestions", []),
                similarity_scores={r["tool_slug"]: r["score"] for r in filtered_results["results"]}
            )
            
            # Save assistant response
            conversation_service.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=filtered_results["message"]
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Ambiguous results (completed in {elapsed_time:.2f}s)")
            
            return WorkflowResponse(
                conversation_id=conversation_id,
                workflow=None,
                tools_used=[],
                confidence_score=filtered_results["results"][0]["score"],
                status="ambiguous",
                message=filtered_results["message"],
                suggestions=filtered_results.get("suggestions", [])
            )
        
        # Step 5: Generate workflow with Claude
        try:
            workflow_json = claude_service.generate_workflow(
                user_query=request.query,
                retrieved_tools=filtered_results["results"],
                conversation_history=conversation_history
            )
            logger.info(f"Generated workflow with {len(workflow_json.get('nodes', []))} nodes")
        except BedrockException as e:
            logger.error(f"Claude workflow generation failed: {e}")
            
            # Check if rate limit error
            if "Rate limit" in str(e) or "ThrottlingException" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again in a few seconds."
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Workflow generator unavailable: {str(e)}"
                )
        except ValueError as e:
            logger.error(f"Workflow validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid workflow generated: {str(e)}"
            )
        
        # Step 6: Save to database
        try:
            # Extract tools used
            tools_used = list(set([
                result["tool_slug"] 
                for result in filtered_results["results"]
            ]))
            
            # Save user message
            conversation_service.save_message(
                conversation_id=conversation_id,
                role="user",
                content=request.query,
                tools_retrieved=tools_used,
                similarity_scores={r["tool_slug"]: r["score"] for r in filtered_results["results"][:3]}
            )
            
            # Save assistant response
            conversation_service.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content="Generated workflow successfully"
            )
            
            # Save workflow
            conversation_service.save_workflow(
                conversation_id=conversation_id,
                workflow_json=workflow_json
            )
            
            logger.info(f"Saved workflow to database")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save workflow: {str(e)}"
            )
        
        # Step 7: Return response
        elapsed_time = time.time() - start_time
        logger.info(f"Workflow creation completed in {elapsed_time:.2f}s")
        
        return WorkflowResponse(
            conversation_id=conversation_id,
            workflow=workflow_json,
            tools_used=tools_used,
            confidence_score=filtered_results.get("top_score", 1.0),
            status="confident",
            message=None
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error in workflow creation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post("/edit", response_model=WorkflowResponse)
async def edit_workflow(
    request: EditWorkflowRequest,
    db: Session = Depends(get_db_dependency)
):
    """
    Edit an existing workflow.
    
    Args:
        request: EditWorkflowRequest with conversation_id and edit instruction
        db: Database session
        
    Returns:
        WorkflowResponse with updated workflow
    """
    start_time = time.time()
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        qdrant_service = QdrantService()
        claude_service = ClaudeService()
        conversation_service = ConversationService(db)
        
        logger.info(f"Workflow edit request: '{request.edit_instruction[:100]}...'")
        
        # Step 1: Load conversation and current workflow
        conversation = conversation_service.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {request.conversation_id}"
            )
        
        current_workflow = conversation_service.get_current_workflow(request.conversation_id)
        if not current_workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No workflow found to edit. Create a workflow first."
            )
        
        logger.info(f"Loaded current workflow (version: {current_workflow.get('version', 1)})")
        
        # Step 2: Generate embedding for edit instruction
        try:
            query_embedding = embedding_service.generate_embedding(
                request.edit_instruction,
                input_type="query"
            )
        except VoyageAIException as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Embedding service unavailable: {str(e)}"
            )
        
        # Step 3: Search for relevant tools (for the edit)
        try:
            search_results = qdrant_service.search_tools(query_embedding)
            logger.info(f"Retrieved {len(search_results)} tools for edit")
        except QdrantException as e:
            logger.error(f"Qdrant search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Search service unavailable: {str(e)}"
            )
        
        # Step 4: Generate updated workflow
        try:
            updated_workflow = claude_service.generate_workflow_edit(
                current_workflow=current_workflow,
                edit_instruction=request.edit_instruction,
                retrieved_tools=search_results
            )
            logger.info(f"Generated updated workflow")
        except BedrockException as e:
            logger.error(f"Claude edit failed: {e}")
            
            if "Rate limit" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again in a few seconds."
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Workflow editor unavailable: {str(e)}"
                )
        
        # Step 5: Save to database
        try:
            # Save user message
            conversation_service.save_message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.edit_instruction
            )
            
            # Save assistant response
            conversation_service.save_message(
                conversation_id=request.conversation_id,
                role="assistant",
                content="Updated workflow successfully"
            )
            
            # Update workflow
            conversation_service.save_workflow(
                conversation_id=request.conversation_id,
                workflow_json=updated_workflow
            )
            
            logger.info(f"Saved updated workflow")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save workflow: {str(e)}"
            )
        
        # Step 6: Return response
        tools_used = list(set([r["tool_slug"] for r in search_results[:3]]))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Workflow edit completed in {elapsed_time:.2f}s")
        
        return WorkflowResponse(
            conversation_id=request.conversation_id,
            workflow=updated_workflow,
            tools_used=tools_used,
            confidence_score=search_results[0]["score"] if search_results else 1.0,
            status="confident",
            message=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in workflow edit: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/conversation/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db_dependency)
):
    """
    Get conversation details with history and workflow.
    
    Args:
        conversation_id: Conversation UUID
        db: Database session
        
    Returns:
        ConversationResponse with messages and workflow
    """
    try:
        conversation_service = ConversationService(db)
        
        # Get conversation history
        history_data = conversation_service.get_conversation_history(conversation_id)
        
        if not history_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}"
            )
        
        # Get current workflow
        workflow = conversation_service.get_current_workflow(conversation_id)
        
        conversation = history_data["conversation"]
        
        return ConversationResponse(
            conversation_id=conversation_id,
            messages=history_data["messages"],
            workflow=workflow,
            summary=history_data["summary"],
            created_at=conversation.created_at.isoformat(),
            message_count=history_data["total_messages"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}", response_model=DeleteResponse)
async def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db_dependency)
):
    """
    Delete a conversation (soft delete).
    
    Args:
        conversation_id: Conversation UUID
        db: Database session
        
    Returns:
        DeleteResponse with success status
    """
    try:
        conversation_service = ConversationService(db)
        
        success = conversation_service.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}"
            )
        
        logger.info(f"Deleted conversation: {conversation_id}")
        
        return DeleteResponse(
            success=True,
            message="Conversation deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )