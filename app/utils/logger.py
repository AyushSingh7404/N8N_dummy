"""
Logging configuration for the application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from pythonjsonlogger import jsonlogger
from config import get_settings


def setup_logging():
    """
    Setup application-wide logging configuration.
    """
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Create formatters
    # Console formatter (human-readable)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File formatter (JSON for structured logging)
    json_formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # File handler (daily rotation)
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(json_formatter)
    
    # Error file handler (only errors)
    error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    
    # Log startup
    root_logger.info("=" * 60)
    root_logger.info("Logging initialized")
    root_logger.info(f"Log level: {settings.log_level}")
    root_logger.info(f"Log file: {log_file}")
    root_logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_rag_query(logger: logging.Logger, query: str, results: list, time_taken: float):
    """
    Log RAG query details.
    
    Args:
        logger: Logger instance
        query: User query
        results: Search results
        time_taken: Time in seconds
    """
    if results:
        top_result = results[0]
        logger.info(
            f"RAG Query: '{query[:100]}...' | "
            f"Top: {top_result.get('tool_display_name')} - {top_result.get('operation_display_name')} "
            f"(score: {top_result.get('score', 0):.4f}) | "
            f"Time: {time_taken*1000:.0f}ms"
        )
    else:
        logger.info(
            f"RAG Query: '{query[:100]}...' | No results | Time: {time_taken*1000:.0f}ms"
        )


def log_workflow_generation(
    logger: logging.Logger, 
    query: str, 
    workflow: dict, 
    time_taken: float
):
    """
    Log workflow generation details.
    
    Args:
        logger: Logger instance
        query: User query
        workflow: Generated workflow JSON
        time_taken: Time in seconds
    """
    node_count = len(workflow.get("nodes", []))
    logger.info(
        f"Workflow Generated: '{query[:100]}...' | "
        f"Nodes: {node_count} | "
        f"Time: {time_taken*1000:.0f}ms"
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: dict
):
    """
    Log error with additional context.
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Dictionary with contextual information
    """
    logger.error(
        f"ERROR: {type(error).__name__} - {str(error)} | Context: {context}",
        exc_info=True
    )