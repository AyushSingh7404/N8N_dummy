"""
Application entry point.
Run with: python run.py or uvicorn run:app --reload
"""

import uvicorn
from app import create_app

# Create app instance
app = create_app()

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        # reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )