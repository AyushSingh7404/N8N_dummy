"""
Test RAG pipeline manually.
Tests the complete flow: Query → Embedding → Search → Results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_settings
from app.services.embedding_service import EmbeddingService, VoyageAIException
from app.services.qdrant_service import QdrantService, QdrantException


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)
    
    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Initialize services
    try:
        embedding_service = EmbeddingService()
        qdrant_service = QdrantService()
        print("✓ Services initialized")
    except Exception as e:
        print(f"✗ Service initialization failed: {e}")
        return False
    
    # Test queries
    test_queries = [
        "send an email",
        "post message to slack",
        "upload file to google drive",
        "create spreadsheet",
        "send discord notification"
    ]
    
    print(f"\nTesting {len(test_queries)} queries...")
    print("=" * 60)
    
    all_passed = True
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 60)
        
        # Step 1: Generate embedding
        try:
            embedding = embedding_service.generate_embedding(query, input_type="query")
            print(f"   ✓ Generated embedding: {len(embedding)} dimensions")
        except VoyageAIException as e:
            print(f"   ✗ Embedding generation failed: {e}")
            all_passed = False
            continue
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            all_passed = False
            continue
        
        # Step 2: Search Qdrant
        try:
            results = qdrant_service.search_tools(embedding, top_k=3)
            print(f"   ✓ Retrieved {len(results)} tools from Qdrant")
        except QdrantException as e:
            print(f"   ✗ Qdrant search failed: {e}")
            all_passed = False
            continue
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            all_passed = False
            continue
        
        # Step 3: Display results
        if results:
            print(f"\n   Top 3 Results:")
            for j, result in enumerate(results[:3], 1):
                print(f"   {j}. {result['tool_display_name']} - {result['operation_display_name']}")
                print(f"      Score: {result['score']:.4f}")
                print(f"      Category: {result['category']}")
        else:
            print(f"   ⚠ No results found")
            all_passed = False
        
        # Step 4: Filter by threshold
        filtered = qdrant_service.filter_by_similarity_threshold(results)
        print(f"\n   Status: {filtered['status']}")
        
        if filtered['status'] == 'confident':
            print(f"   ✓ Confident match (score: {filtered.get('top_score', 0):.4f})")
        elif filtered['status'] == 'ambiguous':
            print(f"   ⚠ Ambiguous - multiple matches")
        elif filtered['status'] == 'no_match':
            print(f"   ✗ No match found")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All RAG pipeline tests passed!")
    else:
        print("⚠ Some tests failed - check output above")
    print("=" * 60)
    
    return all_passed


def test_specific_query(query: str):
    """
    Test a specific query interactively.
    
    Args:
        query: User query to test
    """
    print("=" * 60)
    print(f"Testing Query: '{query}'")
    print("=" * 60)
    
    # Load settings
    settings = load_settings()
    
    # Initialize services
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    
    # Generate embedding
    print("\n1. Generating embedding...")
    embedding = embedding_service.generate_embedding(query, input_type="query")
    print(f"   ✓ Embedding generated: {len(embedding)} dimensions")
    
    # Search
    print("\n2. Searching Qdrant...")
    results = qdrant_service.search_tools(embedding, top_k=5)
    print(f"   ✓ Found {len(results)} results")
    
    # Display results
    print("\n3. Results:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['tool_display_name']} - {result['operation_display_name']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Category: {result['category']}")
        print(f"   Type: {result['operation_type']}")
        print(f"   Description: {result['content'][:100]}...")
    
    # Filter by threshold
    print("\n4. Threshold Filtering:")
    print("-" * 60)
    filtered = qdrant_service.filter_by_similarity_threshold(results)
    print(f"Status: {filtered['status']}")
    
    if filtered.get('message'):
        print(f"Message: {filtered['message']}")
    
    if filtered.get('suggestions'):
        print(f"Suggestions: {', '.join(filtered['suggestions'])}")
    
    print("=" * 60)


def main():
    """Main test logic."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG pipeline")
    parser.add_argument(
        "--query",
        type=str,
        help="Test a specific query"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test queries"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Test specific query
        test_specific_query(args.query)
    elif args.all:
        # Run all tests
        test_rag_pipeline()
    else:
        # Default: run all tests
        test_rag_pipeline()


if __name__ == "__main__":
    main()