"""
Tests for ClaudeService.
"""

import pytest
import json
from app.services.claude_service import ClaudeService, BedrockException


class TestClaudeService:
    """Tests for Claude workflow generation."""
    
    def test_generate_workflow_success(self, mocker, sample_tools_retrieved):
        """Test successful workflow generation."""
        # Mock Bedrock client
        mock_client = mocker.MagicMock()
        mock_response = {
            'body': mocker.MagicMock()
        }
        
        workflow_json = {
            "nodes": [{"id": "node1", "type": "gmail.send-email"}],
            "connections": {}
        }
        
        response_body = {
            'content': [{'text': json.dumps(workflow_json)}]
        }
        
        mock_response['body'].read.return_value = json.dumps(response_body)
        mock_client.invoke_model.return_value = mock_response
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        result = service.generate_workflow(
            "Send email",
            sample_tools_retrieved
        )
        
        assert result == workflow_json
        assert "nodes" in result
        assert "connections" in result
    
    def test_generate_workflow_with_markdown_fences(self, mocker, sample_tools_retrieved):
        """Test handling of markdown-wrapped JSON."""
        mock_client = mocker.MagicMock()
        mock_response = {'body': mocker.MagicMock()}
        
        workflow_json = {
            "nodes": [{"id": "node1", "type": "gmail.send-email"}],
            "connections": {}
        }
        
        # Wrap JSON in markdown fences
        text_with_fences = f"```json\n{json.dumps(workflow_json)}\n```"
        
        response_body = {
            'content': [{'text': text_with_fences}]
        }
        
        mock_response['body'].read.return_value = json.dumps(response_body)
        mock_client.invoke_model.return_value = mock_response
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        result = service.generate_workflow(
            "Send email",
            sample_tools_retrieved
        )
        
        assert result == workflow_json
    
    def test_generate_workflow_invalid_json_retry(self, mocker, sample_tools_retrieved):
        """Test retry logic when JSON is invalid."""
        mock_client = mocker.MagicMock()
        
        # First two attempts return invalid JSON, third succeeds
        valid_workflow = {
            "nodes": [{"id": "node1", "type": "gmail.send-email"}],
            "connections": {}
        }
        
        responses = [
            # Attempt 1: Invalid JSON
            json.dumps({'content': [{'text': 'invalid json {'}]}),
            # Attempt 2: Still invalid
            json.dumps({'content': [{'text': '{incomplete'}]}),
            # Attempt 3: Valid
            json.dumps({'content': [{'text': json.dumps(valid_workflow)}]})
        ]
        
        mock_client.invoke_model.side_effect = [
            {'body': mocker.MagicMock(read=mocker.MagicMock(return_value=r))}
            for r in responses
        ]
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        result = service.generate_workflow(
            "Send email",
            sample_tools_retrieved
        )
        
        assert result == valid_workflow
        assert mock_client.invoke_model.call_count == 3
    
    def test_generate_workflow_max_retries_exceeded(self, mocker, sample_tools_retrieved):
        """Test failure after max retries."""
        mock_client = mocker.MagicMock()
        
        # All attempts return invalid JSON
        invalid_response = json.dumps({'content': [{'text': 'invalid json'}]})
        mock_client.invoke_model.return_value = {
            'body': mocker.MagicMock(read=mocker.MagicMock(return_value=invalid_response))
        }
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        
        with pytest.raises(BedrockException, match="Failed to parse valid JSON"):
            service.generate_workflow("Send email", sample_tools_retrieved)
        
        assert mock_client.invoke_model.call_count == 3
    
    def test_generate_workflow_validation_error(self, mocker, sample_tools_retrieved):
        """Test workflow validation catches invalid structure."""
        mock_client = mocker.MagicMock()
        
        # Return JSON without required fields
        invalid_workflow = {"nodes": []}  # Missing connections
        
        response_body = {'content': [{'text': json.dumps(invalid_workflow)}]}
        mock_client.invoke_model.return_value = {
            'body': mocker.MagicMock(read=mocker.MagicMock(
                return_value=json.dumps(response_body)
            ))
        }
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        
        with pytest.raises(BedrockException):
            service.generate_workflow("Send email", sample_tools_retrieved)
    
    def test_generate_workflow_with_conversation_history(self, mocker, sample_tools_retrieved):
        """Test workflow generation with conversation history."""
        mock_client = mocker.MagicMock()
        
        workflow_json = {
            "nodes": [{"id": "node1", "type": "gmail.send-email"}],
            "connections": {}
        }
        
        response_body = {'content': [{'text': json.dumps(workflow_json)}]}
        mock_client.invoke_model.return_value = {
            'body': mocker.MagicMock(read=mocker.MagicMock(
                return_value=json.dumps(response_body)
            ))
        }
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        
        history = [
            {"role": "user", "content": "Send email"},
            {"role": "assistant", "content": "I'll create that workflow"}
        ]
        
        result = service.generate_workflow(
            "Now change to Slack",
            sample_tools_retrieved,
            conversation_history=history
        )
        
        assert result == workflow_json
        
        # Verify history was included in messages
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args.kwargs['body'])
        assert len(body['messages']) == 3  # 2 history + 1 new
    
    def test_generate_workflow_edit_success(self, mocker, sample_tools_retrieved):
        """Test workflow editing."""
        mock_client = mocker.MagicMock()
        
        current_workflow = {
            "nodes": [{"id": "node1", "type": "gmail.send-email"}],
            "connections": {}
        }
        
        updated_workflow = {
            "nodes": [{"id": "node1", "type": "slack.send-message"}],
            "connections": {}
        }
        
        response_body = {'content': [{'text': json.dumps(updated_workflow)}]}
        mock_client.invoke_model.return_value = {
            'body': mocker.MagicMock(read=mocker.MagicMock(
                return_value=json.dumps(response_body)
            ))
        }
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        result = service.generate_workflow_edit(
            current_workflow,
            "Change to Slack",
            sample_tools_retrieved
        )
        
        assert result == updated_workflow
    
    def test_generate_summary_success(self, mocker):
        """Test conversation summary generation."""
        mock_client = mocker.MagicMock()
        
        summary_text = "User wants to send emails via Gmail"
        response_body = {'content': [{'text': summary_text}]}
        
        mock_client.invoke_model.return_value = {
            'body': mocker.MagicMock(read=mocker.MagicMock(
                return_value=json.dumps(response_body)
            ))
        }
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        
        messages = [
            {"role": "user", "content": "Send email"},
            {"role": "assistant", "content": "I'll help with that"}
        ]
        
        summary = service.generate_summary(messages)
        
        assert summary == summary_text
    
    def test_generate_summary_failure_returns_empty(self, mocker):
        """Test summary generation failure doesn't block flow."""
        mock_client = mocker.MagicMock()
        mock_client.invoke_model.side_effect = Exception("API error")
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        
        messages = [{"role": "user", "content": "Test"}]
        summary = service.generate_summary(messages)
        
        # Should return empty string on failure
        assert summary == ""
    
    def test_bedrock_rate_limit_error(self, mocker, sample_tools_retrieved):
        """Test handling of rate limit errors."""
        from botocore.exceptions import ClientError
        
        mock_client = mocker.MagicMock()
        
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate exceeded'
            }
        }
        
        mock_client.invoke_model.side_effect = ClientError(
            error_response,
            'InvokeModel'
        )
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        
        with pytest.raises(BedrockException, match="Rate limit exceeded"):
            service.generate_workflow("Send email", sample_tools_retrieved)
    
    def test_bedrock_validation_error(self, mocker, sample_tools_retrieved):
        """Test handling of validation errors."""
        from botocore.exceptions import ClientError
        
        mock_client = mocker.MagicMock()
        
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid request'
            }
        }
        
        mock_client.invoke_model.side_effect = ClientError(
            error_response,
            'InvokeModel'
        )
        
        mocker.patch('boto3.client', return_value=mock_client)
        
        service = ClaudeService()
        
        with pytest.raises(BedrockException, match="Invalid request"):
            service.generate_workflow("Send email", sample_tools_retrieved)