"""
Configuration management for the RAG Tool Retrieval System.
Loads and validates environment variables.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # AWS Bedrock Configuration
    aws_access_key_id: str = Field(..., description="AWS Access Key ID")
    aws_secret_access_key: str = Field(..., description="AWS Secret Access Key")
    aws_region: str = Field(default="ap-south-1", description="AWS Region")
    claude_model_id: str = Field(
        default="anthropic.claude-sonnet-4-20250514-v1:0",
        description="Claude Model ID"
    )
    
    # Voyage AI Configuration
    voyage_ai_key: str = Field(..., description="Voyage AI API Key")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection_name: str = Field(
        default="tool_operations",
        description="Qdrant collection name"
    )
    
    # Database Configuration
    sqlite_db_path: str = Field(
        default="./data/workflows.db",
        description="SQLite database path"
    )
    
    # RAG Configuration
    top_k_tools: int = Field(default=5, description="Number of tools to retrieve")
    similarity_threshold_high: float = Field(
        default=0.7,
        description="High confidence similarity threshold"
    )
    similarity_threshold_low: float = Field(
        default=0.5,
        description="Low confidence similarity threshold (no match)"
    )
    ambiguity_threshold: float = Field(
        default=0.15,
        description="Difference threshold for ambiguous results"
    )
    
    # Application Configuration
    flask_env: str = Field(default="development", description="Environment")
    flask_debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Tools Configuration
    tools_json_path: str = Field(
        default="./data/tools/tools_metadata.json",
        description="Path to tools metadata JSON"
    )
    
    # Embedding Configuration
    embedding_dimension: int = Field(default=1024, description="Voyage-code-3 dimension")
    voyage_model: str = Field(default="voyage-code-3", description="Voyage AI model")
    
    # Claude Configuration
    claude_max_tokens: int = Field(default=4000, description="Max tokens for Claude")
    claude_temperature: float = Field(default=0.3, description="Claude temperature")
    
    @validator('sqlite_db_path')
    def validate_db_path(cls, v):
        """Ensure database directory exists."""
        db_path = Path(v)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('tools_json_path')
    def validate_tools_path(cls, v):
        """Ensure tools directory exists."""
        tools_path = Path(v)
        tools_path.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings: Optional[Settings] = None


def load_settings() -> Settings:
    """
    Load and validate settings from environment.
    
    Returns:
        Settings: Validated settings object
        
    Raises:
        ValueError: If required environment variables are missing
    """
    global settings
    
    if settings is None:
        try:
            settings = Settings()
            _validate_critical_settings(settings)
            _log_settings(settings)
        except Exception as e:
            raise ValueError(f"Configuration error: {str(e)}")
    
    return settings


def _validate_critical_settings(settings: Settings) -> None:
    """
    Validate that all critical settings are present.
    
    Args:
        settings: Settings object to validate
        
    Raises:
        ValueError: If critical settings are missing
    """
    critical_fields = [
        'aws_access_key_id',
        'aws_secret_access_key',
        'voyage_ai_key'
    ]
    
    missing = []
    for field in critical_fields:
        value = getattr(settings, field, None)
        if not value or value == f"your_{field}_here":
            missing.append(field.upper())
    
    if missing:
        raise ValueError(
            f"Missing critical environment variables: {', '.join(missing)}. "
            f"Please set them in your .env file."
        )


def _log_settings(settings: Settings) -> None:
    """
    Log loaded configuration (masking sensitive values).
    
    Args:
        settings: Settings object to log
    """
    print("=" * 60)
    print("Configuration Loaded Successfully")
    print("=" * 60)
    print(f"Environment: {settings.flask_env}")
    print(f"Log Level: {settings.log_level}")
    print(f"AWS Region: {settings.aws_region}")
    print(f"Claude Model: {settings.claude_model_id}")
    print(f"Voyage Model: {settings.voyage_model}")
    print(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"Database: {settings.sqlite_db_path}")
    print(f"Tools JSON: {settings.tools_json_path}")
    print(f"Top-K Tools: {settings.top_k_tools}")
    print(f"Similarity Thresholds: High={settings.similarity_threshold_high}, Low={settings.similarity_threshold_low}")
    print("=" * 60)


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings: Global settings object
    """
    if settings is None:
        return load_settings()
    return settings