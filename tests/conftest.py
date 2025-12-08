"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from app.models.base import Base, get_db_dependency
from config import Settings, get_settings


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    return Settings(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        aws_region="us-east-1",
        claude_model_id="test-model",
        voyage_ai_key="test_voyage_key",
        qdrant_host="localhost",
        qdrant_port=6333,
        sqlite_db_path=":memory:",  # In-memory database for tests
        log_level="ERROR"  # Reduce noise in tests
    )


@pytest.fixture(scope="function")
def test_db():
    """Create a test database for each test."""
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db, test_settings):
    """Create test client with overridden dependencies."""
    app = create_app()
    
    # Override database dependency
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    # Override settings
    def override_get_settings():
        return test_settings
    
    app.dependency_overrides[get_db_dependency] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding (1024 dimensions)."""
    return [0.1] * 1024


@pytest.fixture
def sample_tools_retrieved():
    """Sample tools retrieved from Qdrant."""
    return [
        {
            "id": "gmail_send-email",
            "score": 0.85,
            "tool_name": "gmail",
            "tool_slug": "gmail",
            "tool_display_name": "Gmail",
            "operation_name": "send_email",
            "operation_slug": "send-email",
            "operation_display_name": "Send Email",
            "category": "email",
            "operation_type": "action",
            "content": "Send an email through Gmail...",
            "required_fields": ["to", "subject", "body"],
            "tags": ["email", "google"],
            "auth_required": True
        },
        {
            "id": "slack_send-message",
            "score": 0.65,
            "tool_name": "slack",
            "tool_slug": "slack",
            "tool_display_name": "Slack",
            "operation_name": "send_message",
            "operation_slug": "send-message",
            "operation_display_name": "Send Message",
            "category": "communication",
            "operation_type": "action",
            "content": "Send a message to Slack...",
            "required_fields": ["channel", "text"],
            "tags": ["messaging", "team"],
            "auth_required": True
        }
    ]


@pytest.fixture
def sample_workflow():
    """Sample workflow JSON."""
    return {
        "nodes": [
            {
                "id": "node1",
                "type": "gmail.send-email",
                "displayName": "Send Email",
                "parameters": {
                    "to": "user@example.com",
                    "subject": "Test",
                    "body": "Test email"
                }
            }
        ],
        "connections": {}
    }


@pytest.fixture
def mock_embedding_service(mocker, sample_query_embedding):
    """Mock EmbeddingService."""
    mock = mocker.patch("app.services.embedding_service.EmbeddingService")
    mock.return_value.generate_embedding.return_value = sample_query_embedding
    return mock


@pytest.fixture
def mock_qdrant_service(mocker, sample_tools_retrieved):
    """Mock QdrantService."""
    mock = mocker.patch("app.services.qdrant_service.QdrantService")
    mock.return_value.search_tools.return_value = sample_tools_retrieved
    mock.return_value.filter_by_similarity_threshold.return_value = {
        "status": "confident",
        "results": sample_tools_retrieved,
        "top_score": 0.85
    }
    return mock


@pytest.fixture
def mock_claude_service(mocker, sample_workflow):
    """Mock ClaudeService."""
    mock = mocker.patch("app.services.claude_service.ClaudeService")
    mock.return_value.generate_workflow.return_value = sample_workflow
    mock.return_value.generate_workflow_edit.return_value = sample_workflow
    return mock