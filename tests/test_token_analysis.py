import unittest
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import io
import sys
from contextlib import redirect_stdout
import os
import sys

# Add parent directory to path so we can import token_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from token_analysis.py
from token_analysis import (
    get_tokenizer,
    extract_by_criteria,
    extract_tool_data,
    count_tool_tokens,
    count_tools_for_schema,
    analyze_logs,
    print_analysis,
    read_jsonl_file
)

class TestTokenAnalysis(unittest.TestCase):
    def setUp(self):
        # Mock tokenizer that returns predictable token counts
        self.mock_tokenizer = lambda text: len(text.split())
        
        # Sample tool data for testing
        self.sample_tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state"
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    }
                }
            }
        ]
        
        # Sample message content with tool requests and responses
        self.sample_content = [
            {"type": "text", "text": "Hello world"},
            {
                "type": "toolRequest",
                "id": "tool-1",
                "toolCall": {
                    "name": "get_weather",
                    "arguments": {"location": "San Francisco", "unit": "celsius"}
                }
            },
            {
                "type": "toolResponse",
                "id": "tool-1",
                "content": [{"type": "text", "text": "The weather is sunny and 22°C"}]
            }
        ]
        
        # Sample session logs
        self.sample_logs = [
            # Summary
            {"tools": self.sample_tools},
            # User message
            {
                "role": "user",
                "created": 1650000000,
                "content": [{"type": "text", "text": "What's the weather in San Francisco?"}]
            },
            # Assistant message with tool request
            {
                "role": "assistant",
                "created": 1650000010,
                "content": [
                    {"type": "text", "text": "I'll check the weather for you."},
                    {
                        "type": "toolRequest",
                        "id": "tool-1",
                        "toolCall": {
                            "name": "get_weather",
                            "arguments": {"location": "San Francisco", "unit": "celsius"}
                        }
                    }
                ]
            },
            # User message with tool response
            {
                "role": "user",
                "created": 1650000020,
                "content": [
                    {
                        "type": "toolResponse",
                        "id": "tool-1",
                        "content": [{"type": "text", "text": "The weather is sunny and 22°C"}]
                    }
                ]
            },
            # Assistant final response
            {
                "role": "assistant",
                "created": 1650000030,
                "content": [{"type": "text", "text": "The weather in San Francisco is currently sunny with a temperature of 22°C."}]
            }
        ]

    @patch('tiktoken.get_encoding')
    def test_get_tokenizer(self, mock_get_encoding):
        # Setup mock
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # 3 tokens
        mock_get_encoding.return_value = mock_encoding

        # Test the tokenizer
        tokenizer = get_tokenizer('tiktoken')
        result = tokenizer("test text")

        # Verify
        self.assertEqual(result, 3)
        mock_get_encoding.assert_called_once_with("cl100k_base")

    def test_extract_by_criteria(self):
        # Test extracting text from mixed content
        result = extract_by_criteria(
            self.sample_content,
            match_fn=lambda x: isinstance(x, dict) and x.get("type") == "text",
            value_fn=lambda x: x.get("text", "")
        )
        
        self.assertEqual(result, "Hello world The weather is sunny and 22°C")

    def test_extract_tool_data(self):
        # Load the sample session data
        sample_file = os.path.join('tests', 'test_data', 'sample_session.jsonl')
        with open(sample_file, 'r') as f:
            session_data = [json.loads(line) for line in f if line.strip()]

        # Extract the assistant message with tool request
        assistant_msg = next(msg for msg in session_data if msg.get('role') == 'assistant' and any(
            item.get('type') == 'toolRequest' for item in msg.get('content', []) if isinstance(item, dict)
        ))

        # Test tool request extraction
        tool_requests = extract_tool_data(assistant_msg['content'], "toolRequest")
        self.assertEqual(len(tool_requests), 1)
        self.assertEqual(tool_requests[0]["id"], "tool-1")
        self.assertEqual(tool_requests[0]["name"], "get_weather")
        self.assertEqual(tool_requests[0]["arguments"], {"location": "San Francisco", "unit": "celsius"})

        # Extract the user message with tool response
        user_msg = next(msg for msg in session_data if msg.get('role') == 'user' and any(
            item.get('type') == 'toolResponse' for item in msg.get('content', []) if isinstance(item, dict)
        ))

        # Test tool response extraction
        tool_responses = extract_tool_data(user_msg['content'], "toolResponse")
        self.assertEqual(len(tool_responses), 1)
        self.assertEqual(tool_responses[0]["id"], "tool-1")
        self.assertEqual(tool_responses[0]["content"], [{"type": "text", "text": "The weather is sunny and 22°C"}])
        
    def test_count_tool_tokens(self):
        # Test counting tokens for tool requests
        tool_requests = extract_tool_data(self.sample_content, "toolRequest")
        request_tokens = count_tool_tokens(self.mock_tokenizer, tool_requests, is_request=True)
        
        # The format is "{id}:{name}:{arguments_json}" which should have multiple tokens
        self.assertGreater(request_tokens, 0)
        
        # Test counting tokens for tool responses
        tool_responses = extract_tool_data(self.sample_content, "toolResponse")
        response_tokens = count_tool_tokens(self.mock_tokenizer, tool_responses, is_request=False)
        
        # The format is "{id}:{response_text}" which should have multiple tokens
        self.assertGreater(response_tokens, 0)

    def test_count_tools_for_schema(self):
        # Test counting tokens for tool schema
        schema_tokens = count_tools_for_schema(self.mock_tokenizer, self.sample_tools)

        # The schema should have tokens for name, description, properties, etc.
        self.assertGreater(schema_tokens, 0)

        # Test with empty tools list
        empty_schema_tokens = count_tools_for_schema(self.mock_tokenizer, [])
        self.assertEqual(empty_schema_tokens, 0)

        # Create a copy of sample_tools with additional enum values
        import copy
        tools_with_more_enums = copy.deepcopy(self.sample_tools)
        # Add more enum values to the existing enum
        for tool in tools_with_more_enums:
            for prop_name, prop in tool.get('input_schema', {}).get('properties', {}).items():
                if 'enum' in prop:
                    # Add more enum values
                    prop['enum'].extend(['kelvin', 'rankine', 'delisle'])

        # Test with a tool that has more enum values
        enum_schema_tokens = count_tools_for_schema(self.mock_tokenizer, tools_with_more_enums)
        self.assertGreater(enum_schema_tokens, schema_tokens,
                        "Tools with more enum values should have more tokens")

    def test_analyze_logs(self):
        # Test analyzing logs
        with patch('token_analysis.get_tokenizer', return_value=self.mock_tokenizer):
            analyzed = analyze_logs(self.sample_logs, tokenizer_name='tiktoken')
        
        # Should have 4 messages (excluding summary)
        self.assertEqual(len(analyzed), 4)
        
        # First user message should have system overhead added
        self.assertGreater(analyzed[0]['input_tokens'], 3000)  # SYSTEM_PROMPT_OVERHEAD
        
        # Check message types
        self.assertEqual(analyzed[0]['type'], 'user_input')
        self.assertEqual(analyzed[1]['type'], 'agent_output')
        self.assertEqual(analyzed[2]['type'], 'tool_call')
        self.assertEqual(analyzed[3]['type'], 'agent_output')
        
        # Check token counts
        for item in analyzed:
            if item['role'] == 'user':
                self.assertGreaterEqual(item['input_tokens'], 0)
                self.assertEqual(item['output_tokens'], 0)
            else:  # assistant
                self.assertEqual(item['input_tokens'], 0)
                self.assertGreaterEqual(item['output_tokens'], 0)

    def test_print_analysis(self):
        # Create sample analyzed data
        analyzed = [
            {
                'role': 'user',
                'type': 'user_input',
                'created': 1650000000,
                'context_tokens': 0,
                'input_tokens': 3500,  # Including system overhead
                'output_tokens': 0,
                'details': 'What is the weather?'
            },
            {
                'role': 'assistant',
                'type': 'agent_output',
                'created': 1650000010,
                'context_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 50,
                'details': 'I will check the weather for you.'
            }
        ]
        
        # Capture stdout to verify output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            print_analysis(analyzed)
        
        output = captured_output.getvalue()
        
        # Check that key sections are in the output
        self.assertIn("=== Session Details ===", output)
        self.assertIn("=== Session Summary ===", output)
        self.assertIn("=== Token Usage Distribution ===", output)
        
        # Check that metrics are included
        self.assertIn("Total interactions", output)
        self.assertIn("System prompt & tools overhead", output)
        self.assertIn("Total tokens", output)

    @patch('token_analysis.open')
    def test_read_jsonl_file(self, mock_open):
        # Mock file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value = [
            '{"key": "value1"}\n',
            '\n',  # Empty line should be skipped
            '{"key": "value2"}\n'
        ]
        mock_open.return_value = mock_file
        
        # Test reading JSONL file
        result = read_jsonl_file('test.jsonl')
        
        # Should have 2 objects (skipping empty line)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['key'], 'value1')
        self.assertEqual(result[1]['key'], 'value2')

if __name__ == '__main__':
    unittest.main()
