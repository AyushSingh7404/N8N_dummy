"""
Claude service for workflow generation using AWS Bedrock.
"""

import json
import re
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
from config import get_settings


class BedrockException(Exception):
    """Custom exception for AWS Bedrock errors."""
    pass


class ClaudeService:
    """Service for generating workflows using Claude via AWS Bedrock."""
    
    def __init__(self):
        """Initialize Claude service."""
        self.settings = get_settings()
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.settings.aws_region,
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key
        )
        self.model_id = self.settings.claude_model_id
        self.max_retries = 3
    
    def generate_workflow(
        self,
        user_query: str,
        retrieved_tools: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        existing_workflow: Optional[Dict[str, Any]] = None,   # NEW
    ) -> Dict[str, Any]:
        """
        Generate workflow JSON from user query and retrieved tools.
        
        Args:
            user_query: User's natural language request
            retrieved_tools: Tools retrieved from Qdrant
            conversation_history: Optional conversation history
            existing_workflow=existing_workflow,               # NEW
        Returns:
            Dict: Workflow JSON with nodes and connections
            
        Raises:
            BedrockException: If workflow generation fails
        """
        # Build tools context
        tools_context = self._format_tools_context(retrieved_tools)
        
        # Build prompt
        prompt = self._build_workflow_prompt(
            user_query=user_query,
            tools_context=tools_context,
            existing_workflow=existing_workflow        # NEW
        )
        
        # Build messages array
        messages = []
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Call Claude with retry logic
        for attempt in range(self.max_retries):
            try:
                response_text = self._call_bedrock(messages)
                
                # Parse JSON from response
                workflow_json = self._parse_json_response(response_text)
                
                # Validate workflow structure
                self._validate_workflow(workflow_json)
                
                return workflow_json
                
            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    # Retry with stricter prompt
                    print(f"JSON parse error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    messages[-1]["content"] = self._build_stricter_prompt(user_query, tools_context)
                else:
                    raise BedrockException(
                        f"Failed to parse valid JSON after {self.max_retries} attempts. "
                        f"Last response: {response_text[:200]}"
                    )
            except Exception as e:
                if attempt < self.max_retries - 1 and "ThrottlingException" not in str(e):
                    print(f"Claude error (attempt {attempt + 1}/{self.max_retries}): {e}")
                else:
                    raise BedrockException(f"Workflow generation failed: {str(e)}")
    
    def generate_workflow_edit(
        self,
        current_workflow: Dict[str, Any],
        edit_instruction: str,
        retrieved_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate updated workflow based on edit instruction.
        
        Args:
            current_workflow: Current workflow JSON
            edit_instruction: User's edit instruction
            retrieved_tools: Tools retrieved for the edit
            
        Returns:
            Dict: Updated workflow JSON
            
        Raises:
            BedrockException: If edit generation fails
        """
        # Build tools context
        tools_context = self._format_tools_context(retrieved_tools)
        
        # Build edit prompt
        prompt = f"""Current workflow:
{json.dumps(current_workflow, indent=2)}

User wants to: {edit_instruction}

Available tools:
{tools_context}

Output the COMPLETE updated workflow as valid JSON.
Include all nodes and connections.
Output ONLY the JSON, no markdown, no explanations.
"""
        
        # Call Claude
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = self._call_bedrock(messages)
            workflow_json = self._parse_json_response(response_text)
            self._validate_workflow(workflow_json)
            return workflow_json
        except Exception as e:
            raise BedrockException(f"Workflow edit failed: {str(e)}")
    
    def generate_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a summary of conversation messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            str: Summary text
        """
        # Build summary prompt
        messages_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        prompt = f"""Summarize this conversation in 2-3 sentences.
Focus on: user's goal, tools discussed, key decisions made.

Conversation:
{messages_text}

Summary:"""
        
        try:
            response = self._call_bedrock([{"role": "user", "content": prompt}])
            return response.strip()
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return ""  # Return empty string on failure (don't block main flow)
    
    def _call_bedrock(self, messages: List[Dict[str, str]]) -> str:
        """
        Call AWS Bedrock with messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Claude's response text
            
        Raises:
            BedrockException: If API call fails
        """
        try:
            # Build request body
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.settings.claude_max_tokens,
                "temperature": self.settings.claude_temperature,
                "messages": messages
            })
            
            # Invoke model
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract text from content array
            if 'content' not in response_body:
                raise BedrockException("Invalid response format: missing 'content'")
            
            content = response_body['content']
            if not content or len(content) == 0:
                raise BedrockException("Empty response from Claude")
            
            # Get text from first content block
            text = content[0].get('text', '')
            
            return text
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            if error_code == 'ThrottlingException':
                raise BedrockException(f"Rate limit exceeded: {error_message}")
            elif error_code == 'ValidationException':
                raise BedrockException(f"Invalid request: {error_message}")
            else:
                raise BedrockException(f"AWS Bedrock error: {error_code} - {error_message}")
        except Exception as e:
            raise BedrockException(f"Bedrock call failed: {str(e)}")
    
    def _format_tools_context(self, tools: List[Dict[str, Any]]) -> str:
        """Format retrieved tools for Claude context."""
        context_parts = []
        
        for tool in tools:
            context_parts.append(
                f"Tool: {tool['tool_display_name']}\n"
                f"Operation: {tool['operation_display_name']}\n"
                f"Description: {tool['content']}\n"
                f"Score: {tool['score']:.4f}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_workflow_prompt(
    self,
    user_query: str,
    tools_context: str,
    existing_workflow: Optional[Dict[str, Any]] = None,
) -> str:
        """
        Build prompt for workflow generation.

        If existing_workflow is provided, Claude should EDIT that workflow
        instead of creating a completely new one.
        """
        # Base rules that apply in both cases
        common_rules = f"""
    Available tools: {tools_context}

    IMPORTANT:
    - Only use tools from the list above. Do NOT invent new tools.
    - Set node.type as "tool_slug.operation_slug" (e.g., "gmail.send-email").
    - Use the most relevant tools from the list.
    - Create unique node IDs (node1, node2, node3, ...).
    - Fill parameters based on the user's request.
    - If the user mentions specific values (emails, channel names, spreadsheet ranges, etc.),
    include them in parameters.
    - Output ONLY valid JSON. No markdown, no prose, no backticks.
    """

        # Case 1: NEW workflow (no existing_workflow yet)
        if existing_workflow is None:
            return f"""You are a workflow generator.

    {common_rules}

    User request:
    {user_query}

    Generate a NEW workflow JSON with this structure:
    {{
    "nodes": [
        {{
        "id": "node1",
        "type": "tool_slug.operation_slug",
        "displayName": "Operation Display Name",
        "parameters": {{
            "param1": "value1"
        }}
        }}
    ],
    "connections": {{
        "node1": {{"next": "node2"}}
    }}
    }}

    Rules:
    1. Create a complete workflow for the user's request.
    2. Connect nodes in a logical order.
    3. If parallel actions are needed, you may let one node have "next" as a list, e.g. "next": ["node2", "node3"].
    4. Do NOT mention any tools that are not in the available tools list.

    Return ONLY the JSON for the workflow.
    """

        # Case 2: EDIT an existing workflow
        existing_json = json.dumps(existing_workflow, indent=2)

        return f"""You are a workflow editor.

    We already have an existing workflow. Your job is to UPDATE it to satisfy
    the user's new instruction.

    {common_rules}

    Existing workflow JSON:
    {existing_json}

    User's new instruction:
    {user_query}

    Edit rules:
    1. Keep the overall structure and intent of the existing workflow.
    2. Preserve trigger nodes (e.g., webhook/form submission triggers) unless the user explicitly asks to remove them.
    3. Modify, add, or remove nodes ONLY as needed to satisfy the new instruction.
    4. If the user wants something done "in parallel", adjust the connections so that
    the relevant nodes are executed in parallel (for example, one node having
    "next": ["node2", "node3"]).
    5. Keep node IDs stable when possible (if a node keeps the same purpose, keep its id).
    6. If you remove a node, make sure connections remain valid and the workflow is still executable.

    Output:
    - Return the FULL UPDATED workflow as JSON (not just a diff).
    - Do NOT include any text outside the JSON.
    """

    
    def _build_stricter_prompt(self, user_query: str, tools_context: str) -> str:
        """Build stricter prompt for retry attempts."""
        return f"""IMPORTANT: Output ONLY valid JSON. No text before or after. No markdown code blocks.

User request: "{user_query}"

Available tools:
{tools_context}

Output workflow JSON:"""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from Claude's response, handling markdown fences.
        
        Args:
            response: Raw response text
            
        Returns:
            Dict: Parsed JSON
            
        Raises:
            json.JSONDecodeError: If JSON is invalid
        """
        # Strip whitespace
        response = response.strip()
        
        # Remove markdown code fences if present
        if response.startswith("```"):
            # Remove opening fence (```json or ```)
            response = re.sub(r'^```(?:json)?\n?', '', response)
            # Remove closing fence
            response = re.sub(r'\n?```$', '', response)
            response = response.strip()
        
        # Parse JSON
        return json.loads(response)
    
    def _validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """
        Validate workflow JSON structure.
        
        Args:
            workflow: Workflow JSON to validate
            
        Raises:
            ValueError: If workflow structure is invalid
        """
        # Check required fields
        if "nodes" not in workflow:
            raise ValueError("Workflow missing 'nodes' array")
        
        if "connections" not in workflow:
            raise ValueError("Workflow missing 'connections' object")
        
        # Validate nodes
        if not isinstance(workflow["nodes"], list):
            raise ValueError("'nodes' must be an array")
        
        if len(workflow["nodes"]) == 0:
            raise ValueError("Workflow must have at least one node")
        
        # Check node structure
        node_ids = set()
        for i, node in enumerate(workflow["nodes"]):
            if "id" not in node:
                raise ValueError(f"Node at index {i} missing 'id'")
            if "type" not in node:
                raise ValueError(f"Node at index {i} missing 'type'")
            
            # Check for duplicate IDs
            if node["id"] in node_ids:
                raise ValueError(f"Duplicate node ID: {node['id']}")
            node_ids.add(node["id"])
        
        # Validate connections reference valid node IDs
        if not isinstance(workflow["connections"], dict):
            raise ValueError("'connections' must be an object")
        
        for source_id in workflow["connections"].keys():
            if source_id not in node_ids:
                raise ValueError(f"Connection references unknown node: {source_id}")