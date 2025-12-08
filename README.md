# RAG Tool Retrieval System

AI-powered workflow automation tool retrieval system using RAG (Retrieval-Augmented Generation).

## Features

- **Natural Language Workflow Creation**: Create automation workflows using plain English
- **RAG-Based Tool Retrieval**: Semantic search over 5 tools (Gmail, Slack, Discord, Google Sheets, Google Drive)
- **AI Workflow Generation**: Claude Sonnet 4 generates structured workflow JSON
- **Conversation Management**: Multi-turn conversations with history and summarization
- **Workflow Editing**: Edit workflows via chat-based instructions

## Tech Stack

- **FastAPI**: Modern Python web framework
- **Qdrant**: Vector database for semantic search (self-hosted)
- **Voyage AI**: Embedding generation (voyage-code-3 model)
- **AWS Bedrock**: Claude Sonnet 4 for workflow generation
- **SQLite**: Conversation and workflow persistence
- **Pydantic**: Request/response validation

## Project Structure

```
project_root/
├── app/
│   ├── __init__.py           # FastAPI app factory
│   ├── routes/               # API endpoints
│   ├── services/             # Business logic
│   ├── models/               # Database models
│   ├── schemas/              # Pydantic schemas
│   └── utils/                # Utilities
├── data/
│   ├── tools/                # Tools metadata JSON
│   ├── qdrant_data/          # Qdrant storage
│   └── workflows.db          # SQLite database
├── scripts/
│   ├── init_db.py            # Database setup
│   └── load_tools_to_qdrant.py  # Tool ingestion
├── tests/                    # Test suite
├── config.py                 # Configuration management
├── run.py                    # Application entry point
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- AWS Account (with Bedrock access)
- Voyage AI API Key

### 2. Clone and Install

```bash
# Create project structure
bash project_structure.sh

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your keys
nano .env
```

Required environment variables:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-south-1
CLAUDE_MODEL_ID=anthropic.claude-sonnet-4-20250514-v1:0
VOYAGE_AI_KEY=your_voyage_key
```

### 4. Start Qdrant

```bash
# Create data directory
mkdir -p data/qdrant_data

# Start Qdrant container
docker run -d \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/data/qdrant_data:/qdrant/storage" \
  --name qdrant \
  qdrant/qdrant

# Verify Qdrant is running
curl http://localhost:6333/health
```

### 5. Initialize Database

```bash
python scripts/init_db.py
```

Expected output:
```
✓ Database engine initialized
✓ Database tables created successfully
✓ All tables created: conversations, messages, workflow_states
```

### 6. Load Tools into Qdrant

First, place your tools JSON at `data/tools/tools_metadata.json`, then:

```bash
python scripts/load_tools_to_qdrant.py
```

Expected output:
```
✓ Loaded 5 tools from JSON
✓ Created collection 'tool_operations'
✓ Processed batch of 10 operations (Total: 23)
✓ All operations successfully ingested
✓ Search test passed
```

### 7. Run Application

```bash
# Development mode (auto-reload)
python run.py

# Or with uvicorn directly
uvicorn run:app --reload --host 0.0.0.0 --port 8000
```

Application will start at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## API Endpoints

### Create Workflow
```bash
POST /api/workflow/create
Content-Type: application/json

{
  "query": "Send an email when a form is submitted",
  "conversation_id": null  // optional
}
```

Response:
```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "workflow": {
    "nodes": [...],
    "connections": {...}
  },
  "tools_used": ["gmail"],
  "confidence_score": 0.87,
  "status": "confident"
}
```

### Edit Workflow
```bash
POST /api/workflow/edit
Content-Type: application/json

{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "edit_instruction": "Change Gmail to Slack"
}
```

### Get Conversation
```bash
GET /api/workflow/conversation/{conversation_id}
```

### Delete Conversation
```bash
DELETE /api/workflow/conversation/{conversation_id}
```

### Health Check
```bash
GET /health
```

### List Tools
```bash
GET /api/tools
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock httpx

# Run all tests
pytest

# Run specific test file
pytest tests/test_routes.py

# Run with coverage
pytest --cov=app tests/

# Run worst-case scenarios only
pytest tests/test_worst_cases.py -v
```

## Configuration

### Similarity Thresholds

Edit in `.env`:
```
SIMILARITY_THRESHOLD_HIGH=0.7    # Confident match
SIMILARITY_THRESHOLD_LOW=0.5     # No match below this
AMBIGUITY_THRESHOLD=0.15         # Score difference for ambiguity
```

### Conversation History

- Last 5 messages kept in full
- Older messages summarized automatically
- Summary generated after 10+ messages

### Logging

Logs written to:
- `logs/app_YYYYMMDD.log` - All logs (JSON format)
- `logs/errors_YYYYMMDD.log` - Errors only
- Console output (human-readable)

Change log level in `.env`:
```
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## Troubleshooting

### Qdrant Not Starting
```bash
# Check if port is in use
lsof -i :6333

# Check Qdrant logs
docker logs qdrant

# Restart Qdrant
docker restart qdrant
```

### Embedding Generation Fails
- Verify `VOYAGE_AI_KEY` is correct
- Check Voyage AI quota: https://www.voyageai.com/
- Review logs: `tail -f logs/app_*.log`

### Claude Returns Invalid JSON
- Check AWS Bedrock access in `ap-south-1` region
- Verify Claude Sonnet 4 is enabled in your AWS account
- Check rate limits: retry after a few seconds

### Database Issues
```bash
# Reset database
rm data/workflows.db
python scripts/init_db.py
```

### Tool Ingestion Fails
```bash
# Verify tools JSON exists
cat data/tools/tools_metadata.json

# Check Qdrant is running
curl http://localhost:6333/health

# Re-run ingestion
python scripts/load_tools_to_qdrant.py
```

## Performance Optimization

### For Production

1. **Disable auto-reload**:
   ```bash
   uvicorn run:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Use PostgreSQL instead of SQLite**:
   Update `SQLITE_DB_PATH` to PostgreSQL connection string

3. **Enable Qdrant persistence**:
   Already configured via Docker volume mount

4. **Cache embeddings**:
   Implement Redis caching for frequent queries

5. **Rate limiting**:
   Add rate limiting middleware to prevent abuse

## Known Limitations

- SQLite for MVP (use PostgreSQL for production)
- No authentication/authorization (add JWT middleware)
- No rate limiting (add in production)
- Synchronous embedding generation (consider async)
- No workflow execution (only generation)

## Next Steps

1. Add user authentication
2. Implement workflow execution engine
3. Add more tools (100+ integrations)
4. Implement caching layer
5. Add analytics and monitoring
6. Deploy to cloud (AWS/GCP/Azure)

## Support

For issues or questions:
- Check logs: `logs/app_*.log`
- Review API docs: http://localhost:8000/docs
- Test health endpoint: http://localhost:8000/health

## License

MIT License