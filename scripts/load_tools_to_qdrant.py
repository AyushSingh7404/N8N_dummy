"""
Tool ingestion script.
Loads tools from JSON and stores embeddings in Qdrant.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import voyageai


class ToolIngestionPipeline:
    """Pipeline for ingesting tools into Qdrant."""
    
    def __init__(self):
        """Initialize pipeline with settings."""
        self.settings = load_settings()
        self.voyage_client = voyageai.Client(api_key=self.settings.voyage_ai_key)
        self.qdrant_client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port
        )
        self.processed_count = 0
        self.failed_count = 0
    
    def load_tools_json(self) -> List[Dict[str, Any]]:
        """Load tools from JSON file."""
        tools_path = Path(self.settings.tools_json_path)
        
        if not tools_path.exists():
            raise FileNotFoundError(
                f"Tools JSON not found at: {tools_path}\n"
                f"Please ensure the file exists."
            )
        
        with open(tools_path, 'r') as f:
            tools = json.load(f)
        
        print(f"✓ Loaded {len(tools)} tools from JSON")
        return tools
    
    def create_chunk_content(self, tool: Dict, operation: Dict) -> str:
        """
        Create chunk content for embedding.
        
        Args:
            tool: Tool metadata
            operation: Operation metadata
            
        Returns:
            str: Formatted chunk content
        """
        # Extract fields safely
        tool_name = tool.get('displayName', tool.get('name', 'Unknown'))
        op_name = operation.get('displayName', operation.get('name', 'Unknown'))
        description = operation.get('description', '')
        use_cases = operation.get('useCases', [])
        keywords = operation.get('semanticKeywords', [])
        
        # Get required fields
        required_fields = [
            field['name'] 
            for field in operation.get('inputSchema', []) 
            if field.get('required', False)
        ]
        
        # Get optional fields
        optional_fields = [
            field['name'] 
            for field in operation.get('inputSchema', []) 
            if not field.get('required', False)
        ]
        
        # Build content string
        content = f"""Tool: {tool_name}
Operation: {op_name}
Description: {description}

Use this when: {', '.join(use_cases) if use_cases else 'General purpose'}

Common phrases: {', '.join(keywords) if keywords else 'N/A'}

Required inputs: {', '.join(required_fields) if required_fields else 'None'}
Optional inputs: {', '.join(optional_fields) if optional_fields else 'None'}

Category: {tool.get('category', 'general')}
Type: {operation.get('operationType', 'action')}
"""
        return content.strip()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Voyage AI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                result = self.voyage_client.embed(
                    texts=texts,
                    model=self.settings.voyage_model,
                    input_type="document"
                )
                return result.embeddings
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  ⚠ Embedding error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ✗ Embedding failed after {max_retries} attempts: {e}")
                    raise
    
    def setup_qdrant_collection(self):
        """Create or recreate Qdrant collection."""
        collection_name = self.settings.qdrant_collection_name
        
        # Check if collection exists
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                print(f"⚠ Collection '{collection_name}' already exists")
                response = input("Delete and recreate? (y/n): ").strip().lower()
                
                if response == 'y':
                    self.qdrant_client.delete_collection(collection_name)
                    print(f"✓ Deleted existing collection")
                else:
                    print("Aborted.")
                    sys.exit(0)
        except Exception as e:
            print(f"Error checking collection: {e}")
        
        # Create collection
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created collection '{collection_name}'")
        except Exception as e:
            print(f"✗ Error creating collection: {e}")
            raise
    
    def process_tools(self, tools: List[Dict]) -> None:
        """
        Process all tools and operations.
        
        Args:
            tools: List of tool definitions
        """
        total_operations = sum(len(tool.get('operations', [])) for tool in tools)
        print(f"\nProcessing {total_operations} operations from {len(tools)} tools...")
        print("=" * 60)
        
        # Batch processing
        batch_size = 10
        batch_texts = []
        batch_metadata = []
        
        for tool in tools:
            tool_name = tool.get('name', 'unknown')
            operations = tool.get('operations', [])
            
            print(f"\nProcessing tool: {tool.get('displayName', tool_name)} ({len(operations)} operations)")
            
            for operation in operations:
                op_slug = operation.get('slug', operation.get('name', 'unknown'))
                
                try:
                    # Create chunk content
                    content = self.create_chunk_content(tool, operation)
                    
                    # Create metadata
                    metadata = {
                        "tool_name": tool.get('name'),
                        "tool_slug": tool.get('slug'),
                        "tool_display_name": tool.get('displayName'),
                        "operation_name": operation.get('name'),
                        "operation_slug": op_slug,
                        "operation_display_name": operation.get('displayName'),
                        "category": tool.get('category'),
                        "operation_type": operation.get('operationType'),
                        "required_fields": [
                            f['name'] for f in operation.get('inputSchema', []) 
                            if f.get('required', False)
                        ],
                        "tags": tool.get('tags', []),
                        "content": content,
                        "auth_required": tool.get('authConfig', {}).get('type') != 'none'
                    }
                    
                    # Add to batch
                    batch_texts.append(content)
                    batch_metadata.append({
                        'original_id': f"{tool.get('slug', tool_name)}_{op_slug}",
                        'metadata': metadata
                    })
                    
                    # Process batch if full
                    if len(batch_texts) >= batch_size:
                        self._process_batch(batch_texts, batch_metadata)
                        batch_texts = []
                        batch_metadata = []
                    
                except Exception as e:
                    print(f"  ✗ Failed to process {tool_name}.{op_slug}: {e}")
                    self.failed_count += 1
        
        # Process remaining items
        if batch_texts:
            self._process_batch(batch_texts, batch_metadata)
    
    # def _process_batch(self, texts: List[str], metadata_list: List[Dict]) -> None:
    #     """
    #     Process a batch of operations.
        
    #     Args:
    #         texts: List of chunk contents
    #         metadata_list: List of metadata dictionaries
    #     """
    #     try:
    #         # Generate embeddings
    #         embeddings = self.generate_embeddings(texts)
            
    #         # Create points for Qdrant
    #         points = []
    #         for i, (embedding, meta) in enumerate(zip(embeddings, metadata_list)):
    #             # Generate UUID for point ID (Qdrant requirement)
    #             import uuid
    #             point_id = str(uuid.uuid4())
                
    #             # Add original ID to metadata
    #             meta['metadata']['original_id'] = meta['id']
                
    #             point = PointStruct(
    #                 id=point_id,
    #                 vector=embedding,
    #                 payload=meta['metadata']
    #             )
    #             points.append(point)
            
    #         # Upsert to Qdrant
    #         self.qdrant_client.upsert(
    #             collection_name=self.settings.qdrant_collection_name,
    #             points=points
    #         )
            
    #         self.processed_count += len(points)
    #         print(f"  ✓ Processed batch of {len(points)} operations (Total: {self.processed_count})")
            
    #     except Exception as e:
    #         print(f"  ✗ Batch processing failed: {e}")
    #         self.failed_count += len(texts)
    
    def _process_batch(self, texts: List[str], metadata_list: List[Dict]) -> None:
        """
        Process a batch of operations.
        """
        try:
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)

            # Create points for Qdrant
            points = []
            for embedding, meta in zip(embeddings, metadata_list):
                import uuid
                point_id = str(uuid.uuid4())

                # Use the existing original_id we stored earlier
                meta["metadata"]["original_id"] = meta["original_id"]

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=meta["metadata"],
                )
                points.append(point)

            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
            )

            self.processed_count += len(points)
            print(f"  ✓ Processed batch of {len(points)} operations (Total: {self.processed_count})")

        except Exception as e:
            print(f"  ✗ Batch processing failed: {e}")
            self.failed_count += len(texts)

    # def verify_ingestion(self) -> bool:
    #     """
    #     Verify that all operations were ingested.
        
    #     Returns:
    #         bool: True if verification passed
    #     """
    #     collection_name = self.settings.qdrant_collection_name
        
    #     try:
    #         collection_info = self.qdrant_client.get_collection(collection_name)
    #         actual_count = collection_info.points_count
            
    #         print(f"\n{'=' * 60}")
    #         print(f"Verification:")
    #         print(f"  Expected: {self.processed_count} operations")
    #         print(f"  Actual: {actual_count} points in Qdrant")
    #         print(f"  Failed: {self.failed_count} operations")
            
    #         if actual_count == self.processed_count:
    #             print(f"✓ All operations successfully ingested")
    #             return True
    #         else:
    #             print(f"⚠ Mismatch detected!")
    #             return False
            
    #     except Exception as e:
    #         print(f"✗ Verification failed: {e}")
    #         return False
    
    def verify_ingestion(self) -> bool:
        collection_name = self.settings.qdrant_collection_name

        try:
            count_result = self.qdrant_client.count(
                collection_name=collection_name,
                exact=True,
            )
            actual_count = count_result.count

            print(f"\n{'=' * 60}")
            print("Verification:")
            print(f"  Expected: {self.processed_count} operations")
            print(f"  Actual: {actual_count} points in Qdrant")
            print(f"  Failed: {self.failed_count} operations")

            if actual_count == self.processed_count:
                print("✓ All operations successfully ingested")
                return True
            else:
                print("⚠ Mismatch detected!")
                return False

        except Exception as e:
            print(f"✗ Verification failed: {e}")
            return False
    
    def test_search(self) -> None:
        """Test search functionality with a sample query."""
        print(f"\n{'=' * 60}")
        print("Testing search functionality...")
        
        try:
            # Generate test query embedding
            test_query = "send an email"
            result = self.voyage_client.embed(
                texts=[test_query],
                model=self.settings.voyage_model,
                input_type="query"
            )
            query_embedding = result.embeddings[0]
            
            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.settings.qdrant_collection_name,
                query_vector=query_embedding,
                limit=3
            )
            
            print(f"\nQuery: '{test_query}'")
            print(f"Top 3 results:")
            for i, result in enumerate(search_results, 1):
                print(f"  {i}. {result.payload.get('tool_display_name')} - "
                      f"{result.payload.get('operation_display_name')} "
                      f"(score: {result.score:.4f})")
            
            print(f"✓ Search test passed")
            
        except Exception as e:
            print(f"✗ Search test failed: {e}")


def main():
    """Main ingestion logic."""
    print("=" * 60)
    print("Tool Ingestion Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = ToolIngestionPipeline()
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        sys.exit(1)
    
    # Load tools
    try:
        tools = pipeline.load_tools_json()
    except Exception as e:
        print(f"✗ Error loading tools: {e}")
        sys.exit(1)
    
    # Setup Qdrant collection
    try:
        pipeline.setup_qdrant_collection()
    except Exception as e:
        print(f"✗ Error setting up Qdrant: {e}")
        sys.exit(1)
    
    # Process tools
    try:
        start_time = time.time()
        pipeline.process_tools(tools)
        elapsed_time = time.time() - start_time
        print(f"\n✓ Processing complete in {elapsed_time:.2f}s")
    except Exception as e:
        print(f"✗ Processing error: {e}")
        sys.exit(1)
    
    # Verify ingestion
    if not pipeline.verify_ingestion():
        print("⚠ Ingestion verification failed")
    
    # Test search
    pipeline.test_search()
    
    print("=" * 60)
    print("✓ Tool ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()