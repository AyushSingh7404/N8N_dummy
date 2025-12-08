"""
Validation utilities for workflows and data.
"""

import re
from typing import Dict, Any, List, Optional


def is_valid_uuid(uuid_string: str) -> bool:
    """
    Check if string is a valid UUID v4.
    
    Args:
        uuid_string: String to validate
        
    Returns:
        bool: True if valid UUID
    """
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, uuid_string, re.IGNORECASE))


def is_valid_email(email: str) -> bool:
    """
    Check if string is a valid email address.
    
    Args:
        email: Email string to validate
        
    Returns:
        bool: True if valid email
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))


def validate_workflow_structure(workflow: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate workflow JSON structure.
    
    Args:
        workflow: Workflow JSON to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check required top-level fields
    if "nodes" not in workflow:
        return False, "Missing 'nodes' array"
    
    if "connections" not in workflow:
        return False, "Missing 'connections' object"
    
    # Validate nodes
    if not isinstance(workflow["nodes"], list):
        return False, "'nodes' must be an array"
    
    if len(workflow["nodes"]) == 0:
        return False, "Workflow must have at least one node"
    
    # Collect node IDs and validate each node
    node_ids = set()
    for i, node in enumerate(workflow["nodes"]):
        # Check required node fields
        if "id" not in node:
            return False, f"Node at index {i} missing 'id'"
        
        if "type" not in node:
            return False, f"Node at index {i} missing 'type'"
        
        # Check for duplicate IDs
        node_id = node["id"]
        if node_id in node_ids:
            return False, f"Duplicate node ID: {node_id}"
        node_ids.add(node_id)
        
        # Validate node type format (should be "tool.operation")
        if "." not in node.get("type", ""):
            return False, f"Node '{node_id}' has invalid type format (expected 'tool.operation')"
    
    # Validate connections
    if not isinstance(workflow["connections"], dict):
        return False, "'connections' must be an object"
    
    # Check that all connection source IDs exist
    for source_id in workflow["connections"].keys():
        if source_id not in node_ids:
            return False, f"Connection references unknown source node: {source_id}"
        
        # Check target nodes in connections
        connection = workflow["connections"][source_id]
        if isinstance(connection, dict) and "next" in connection:
            target_id = connection["next"]
            if target_id not in node_ids:
                return False, f"Connection references unknown target node: {target_id}"
    
    return True, None


def validate_node_parameters(node: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate node parameters against schema.
    
    Args:
        node: Node with parameters
        schema: Parameter schema definition
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    parameters = node.get("parameters", {})
    
    # Check required fields
    required_fields = schema.get("required_fields", [])
    for field in required_fields:
        if field not in parameters:
            errors.append(f"Missing required parameter: {field}")
    
    # Validate field types (basic validation)
    for field_name, field_value in parameters.items():
        if field_value is None:
            continue
        
        # Check email format for email fields
        if "email" in field_name.lower() and isinstance(field_value, str):
            if not is_valid_email(field_value):
                errors.append(f"Invalid email format: {field_name}")
    
    return len(errors) == 0, errors


def validate_query_length(query: str, min_length: int = 1, max_length: int = 1000) -> tuple[bool, Optional[str]]:
    """
    Validate query string length.
    
    Args:
        query: Query string
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    query_length = len(query.strip())
    
    if query_length < min_length:
        return False, f"Query too short (minimum {min_length} characters)"
    
    if query_length > max_length:
        return False, f"Query too long (maximum {max_length} characters)"
    
    return True, None


def sanitize_query(query: str) -> str:
    """
    Sanitize user query by removing potentially harmful content.
    
    Args:
        query: Raw query string
        
    Returns:
        str: Sanitized query
    """
    # Strip whitespace
    query = query.strip()
    
    # Remove multiple spaces
    query = re.sub(r'\s+', ' ', query)
    
    # Remove control characters
    query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
    
    return query


def detect_workflow_cycles(workflow: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Detect cycles in workflow connections (DAG validation).
    
    Args:
        workflow: Workflow JSON
        
    Returns:
        tuple: (has_cycles, list_of_cycle_paths)
    """
    connections = workflow.get("connections", {})
    
    def has_cycle_from(node_id: str, visited: set, path: List[str]) -> Optional[List[str]]:
        """DFS to detect cycles."""
        if node_id in visited:
            # Found a cycle
            cycle_start = path.index(node_id)
            return path[cycle_start:] + [node_id]
        
        if node_id not in connections:
            return None
        
        visited.add(node_id)
        path.append(node_id)
        
        connection = connections[node_id]
        next_node = None
        
        if isinstance(connection, dict) and "next" in connection:
            next_node = connection["next"]
        elif isinstance(connection, str):
            next_node = connection
        
        if next_node:
            result = has_cycle_from(next_node, visited.copy(), path.copy())
            if result:
                return result
        
        return None
    
    # Check for cycles from each node
    cycles = []
    for node_id in connections.keys():
        cycle = has_cycle_from(node_id, set(), [])
        if cycle:
            cycles.append(" -> ".join(cycle))
    
    return len(cycles) > 0, cycles


def validate_similarity_score(score: float) -> bool:
    """
    Validate that similarity score is in valid range.
    
    Args:
        score: Similarity score
        
    Returns:
        bool: True if valid
    """
    return 0.0 <= score <= 1.0


def validate_embedding_dimension(embedding: List[float], expected_dim: int = 1024) -> tuple[bool, Optional[str]]:
    """
    Validate embedding vector dimension.
    
    Args:
        embedding: Embedding vector
        expected_dim: Expected dimension
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(embedding, list):
        return False, "Embedding must be a list"
    
    if len(embedding) != expected_dim:
        return False, f"Invalid embedding dimension: {len(embedding)} (expected {expected_dim})"
    
    if not all(isinstance(x, (int, float)) for x in embedding):
        return False, "Embedding must contain only numbers"
    
    return True, None