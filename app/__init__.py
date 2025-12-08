"""
FastAPI application factory and configuration.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import time

from config import load_settings
from app.models.base import init_db
from app.routes import workflow_routes, health_routes
from app.utils.logger import setup_logging, get_logger
from app.services.embedding_service import VoyageAIException
from app.services.qdrant_service import QdrantException
from app.services.claude_service import BedrockException


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    # Load settings
    settings = load_settings()
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title="RAG Tool Retrieval API",
        description="AI-powered workflow automation tool retrieval system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this based on your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Include routers
    app.include_router(workflow_routes.router)
    app.include_router(health_routes.router)
    
    logger.info("Routes registered")
    
    # Global exception handlers
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "ValidationError",
                "message": "Invalid request data",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(VoyageAIException)
    async def voyage_exception_handler(request: Request, exc: VoyageAIException):
        """Handle Voyage AI service errors."""
        logger.error(f"Voyage AI error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "EmbeddingServiceUnavailable",
                "message": "Embedding service is temporarily unavailable. Please try again.",
                "details": {"service": "Voyage AI"}
            }
        )
    
    @app.exception_handler(QdrantException)
    async def qdrant_exception_handler(request: Request, exc: QdrantException):
        """Handle Qdrant service errors."""
        logger.error(f"Qdrant error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "SearchServiceUnavailable",
                "message": "Search service is temporarily unavailable. Please try again.",
                "details": {"service": "Qdrant"}
            }
        )
    
    @app.exception_handler(BedrockException)
    async def bedrock_exception_handler(request: Request, exc: BedrockException):
        """Handle AWS Bedrock service errors."""
        logger.error(f"Bedrock error: {exc}")
        
        # Check if rate limit error
        if "Rate limit" in str(exc) or "Throttling" in str(exc):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RateLimitExceeded",
                    "message": "Too many requests. Please wait a few seconds and try again.",
                    "details": {"service": "AWS Bedrock"}
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "WorkflowGeneratorUnavailable",
                    "message": "Workflow generator is temporarily unavailable. Please try again.",
                    "details": {"service": "AWS Bedrock"}
                }
            )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again or contact support.",
                "details": {}
            }
        )
    
    # Request/Response logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests and responses."""
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Duration: {duration*1000:.0f}ms"
        )
        
        return response
    
    logger.info("FastAPI application created successfully")
    
    return app