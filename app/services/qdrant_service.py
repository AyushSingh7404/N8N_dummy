"""
Qdrant service for vector search and tool retrieval.
"""

import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from config import get_settings


class QdrantException(Exception):
    """Custom exception for Qdrant errors."""
    pass


class QdrantService:
    """Service for vector search in Qdrant."""
    
    def __init__(self):
        """Initialize Qdrant service."""
        self.settings = get_settings()
        self.client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port
        )
        self.collection_name = self.settings.qdrant_collection_name
        self.max_retries = 2
        self.retry_delay = 1  # seconds
    
    def search_tools(
        self, 
        query_embedding: List[float], 
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for tools using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return (defaults to settings.top_k_tools)
            metadata_filter: Optional metadata filters
                Examples:
                - {"category": "email"} - Filter by category
                - {"tool_name": {"$in": ["gmail", "slack"]}} - Filter by tool names
                
        Returns:
            List[Dict]: List of search results with metadata and scores
            
        Raises:
            QdrantException: If search fails after retries
        """
        if top_k is None:
            top_k = self.settings.top_k_tools
        
        # Validate embedding dimension
        if len(query_embedding) != self.settings.embedding_dimension:
            raise ValueError(
                f"Invalid embedding dimension: {len(query_embedding)} "
                f"(expected {self.settings.embedding_dimension})"
            )
        
        # Build filter if provided
        query_filter = self._build_filter(metadata_filter) if metadata_filter else None
        
        # Search with retry logic
        for attempt in range(self.max_retries):
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                    with_vectors=False  # Don't return vectors to save bandwidth
                )
                
                # Parse results
                parsed_results = []
                for result in results:
                    parsed_results.append({
                        "id": result.payload.get("original_id", str(result.id)),  # Use original_id from metadata
                        "score": result.score,
                        "tool_name": result.payload.get("tool_name"),
                        "tool_slug": result.payload.get("tool_slug"),
                        "tool_display_name": result.payload.get("tool_display_name"),
                        "operation_name": result.payload.get("operation_name"),
                        "operation_slug": result.payload.get("operation_slug"),
                        "operation_display_name": result.payload.get("operation_display_name"),
                        "category": result.payload.get("category"),
                        "operation_type": result.payload.get("operation_type"),
                        "content": result.payload.get("content"),
                        "required_fields": result.payload.get("required_fields", []),
                        "tags": result.payload.get("tags", []),
                        "auth_required": result.payload.get("auth_required", True)
                    })
                
                return parsed_results
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Qdrant search error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    raise QdrantException(
                        f"Qdrant search failed after {self.max_retries} attempts: {str(e)}"
                    )
    
    # def filter_by_similarity_threshold(
    #     self,
    #     results: List[Dict[str, Any]]
    # ) -> Dict[str, Any]:
    #     """
    #     Filter results based on similarity thresholds with smart tool grouping.

    #     Logic:
    #     - "no_match": top score below low threshold
    #     - "ambiguous": multiple different tools have similar top scores
    #     - "confident": one tool clearly dominates (even if it has multiple operations)

    #     Args:
    #         results: Search results from search_tools()

    #     Returns:
    #         Dict with:
    #             - status: "confident" | "ambiguous" | "no_match"
    #             - results: Filtered results
    #             - message: Optional clarification message
    #             - primary_tool: The primary tool slug (if confident)
    #     """
    #     if not results:
    #         return {
    #             "status": "no_match",
    #             "results": [],
    #             "message": "No tools found matching your request."
    #         }

    #     # Thresholds from settings
    #     threshold_high = self.settings.similarity_threshold_high
    #     threshold_low = self.settings.similarity_threshold_low
    #     ambiguity_threshold = self.settings.ambiguity_threshold

    #     # Global top score
    #     top_score = results[0]["score"]

    #     # If everything is too weak → no match
    #     if top_score < threshold_low:
    #         return {
    #             "status": "no_match",
    #             "results": [],
    #             "message": (
    #                 "No tools found matching your request. "
    #                 "Available categories: email, communication, storage, productivity"
    #             )
    #         }

    #     # --- Group results by tool slug ---
    #     tools_grouped: Dict[str, List[Dict[str, Any]]] = {}
    #     for result in results:
    #         tool_slug = result.get("tool_slug")
    #         if not tool_slug:
    #             # Just in case, skip malformed entries
    #             continue
    #         tools_grouped.setdefault(tool_slug, []).append(result)

    #     # Best (highest-scoring) operation per tool
    #     tool_best_scores: Dict[str, Dict[str, Any]] = {}
    #     for tool_slug, operations in tools_grouped.items():
    #         best_op = max(operations, key=lambda x: x["score"])
    #         tool_best_scores[tool_slug] = best_op

    #     # Sort tools by their best score
    #     sorted_tools = sorted(
    #         tool_best_scores.items(),
    #         key=lambda x: x[1]["score"],
    #         reverse=True
    #     )

    #     # ✅ If there is only ONE distinct tool in the top results,
    #     #    we treat it as CONFIDENT, even if it has multiple operations.
    #     if len(sorted_tools) == 1:
    #         top_tool_slug, top_tool_result = sorted_tools[0]
    #         top_tool_all_operations = tools_grouped[top_tool_slug]
    #         confidence_level = "high" if top_score >= threshold_high else "medium"

    #         return {
    #             "status": "confident",
    #             # Return all operations for that tool so Claude can pick the right one
    #             "results": top_tool_all_operations,
    #             "confidence_level": confidence_level,
    #             "top_score": top_tool_result["score"],
    #             "primary_tool": top_tool_slug,
    #             "tool_operations_count": len(top_tool_all_operations)
    #         }

    #     # --- More than one tool: check ambiguity between best tools ---
    #     top_tool_slug, top_tool_result = sorted_tools[0]
    #     second_tool_slug, second_tool_result = sorted_tools[1]

    #     score_diff = top_tool_result["score"] - second_tool_result["score"]

    #     # If two different tools are too close → ambiguous
    #     if score_diff < ambiguity_threshold:
    #         top_unique_tools = sorted_tools[:3]  # up to top 3 tools
    #         tool_names = [result["tool_display_name"] for _, result in top_unique_tools]
    #         tool_slugs = [slug for slug, _ in top_unique_tools]

    #         return {
    #             "status": "ambiguous",
    #             # Return the best operation for each of the top tools
    #             "results": [result for _, result in top_unique_tools],
    #             "message": (
    #                 "I found multiple tools that could work. "
    #                 f"Did you mean: {', '.join(tool_names)}?"
    #             ),
    #             "suggestions": tool_slugs
    #         }

    #     # Otherwise, one tool clearly dominates → confident
    #     top_tool_all_operations = tools_grouped[top_tool_slug]
    #     confidence_level = "high" if top_score >= threshold_high else "medium"

    #     return {
    #         "status": "confident",
    #         # You can either pass only top_tool_all_operations or all results.
    #         # I recommend only the winner's operations:
    #         "results": top_tool_all_operations,
    #         "confidence_level": confidence_level,
    #         "top_score": top_tool_result["score"],
    #         "primary_tool": top_tool_slug,
    #         "tool_operations_count": len(top_tool_all_operations)
    #     }
    
    def filter_by_similarity_threshold(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Filter results based on similarity thresholds.

        Logic:
        - "no_match": top score below low threshold
        - otherwise: always "confident" and return the top results
          (Claude can decide how to combine tools/operations)

        Args:
            results: Search results from search_tools()

        Returns:
            Dict with:
                - status: "confident" | "no_match"
                - results: Filtered results
                - message: Optional clarification message
                - top_score: score of the best match
        """
        if not results:
            return {
                "status": "no_match",
                "results": [],
                "message": "No tools found matching your request."
            }

        # Thresholds from settings
        threshold_high = self.settings.similarity_threshold_high
        threshold_low = self.settings.similarity_threshold_low

        # Top (global) score
        top_score = results[0]["score"]

        # If top score is too low → treat as no match
        if top_score < threshold_low:
            return {
                "status": "no_match",
                "results": [],
                "message": (
                    "No tools found matching your request. "
                    "Available categories: email, communication, storage, productivity"
                )
            }

        # Otherwise: treat as confident.
        # We do NOT mark anything as "ambiguous" here – instead we let Claude
        # see all retrieved tools and decide how to chain them in the workflow.
        confidence_level = "high" if top_score >= threshold_high else "medium"

        return {
            "status": "confident",
            "results": results,
            "confidence_level": confidence_level,
            "top_score": top_score,
        }

    
    def _build_filter(self, metadata_filter: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from metadata dictionary.
        
        Args:
            metadata_filter: Filter conditions
            
        Returns:
            Filter: Qdrant filter object
        """
        conditions = []
        
        for key, value in metadata_filter.items():
            if isinstance(value, dict):
                # Handle operators like {"$in": [...]}
                if "$in" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value["$in"])
                        )
                    )
            else:
                # Simple equality match
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the Qdrant collection.
        
        Returns:
            Dict with collection stats
            
        Raises:
            QdrantException: If collection info retrieval fails
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "total_operations": collection.points_count,
                "vector_dimension": self.settings.embedding_dimension,
                "status": collection.status
            }
        except Exception as e:
            raise QdrantException(f"Failed to get collection info: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is reachable and collection exists.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False