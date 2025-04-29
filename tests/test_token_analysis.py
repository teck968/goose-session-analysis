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

import token_analysis
from token_analysis import Tokenizer, MessageProcessor, TokenAnalyzer, AnalysisFormatter

class TestTokenAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a simple tokenizer for testing
        self.tokenizer = Tokenizer("tiktoken")

        # Mock the tokenize function to return predictable token counts
        self.tokenizer.tokenize = lambda text: len(text.split())

        # Create processor and analyzer instances
        self.processor = MessageProcessor()
        self.analyzer = TokenAnalyzer("tiktoken")
        self.analyzer.tokenizer.tokenize = self.tokenizer.tokenize

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
                    "value": {
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco", "unit": "celsius"}
                    }
                }
            },
            {
                "type": "toolResponse",
                "id": "tool-1",
                "toolResult": {
                    "value": [{"type": "text", "text": "The weather is sunny and 22°C"}]
                }
            }
        ]

        # Sample session logs
        self.sample_logs = [
            # Metadata
            {
                "working_dir": "/home/user/PROJECTS",
                "description": "Test description",
                "message_count": 4,
                "total_tokens": 100,
                "input_tokens": 25,
                "output_tokens": 75,
                "accumulated_total_tokens": 1250,
                "accumulated_input_tokens": 1000,
                "accumulated_output_tokens": 250,
                "tools": self.sample_tools
            },
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
                            "value": {
                                "name": "get_weather",
                                "arguments": {"location": "San Francisco", "unit": "celsius"}
                            }
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
                        "toolResult": {
                            "value": [{"type": "text", "text": "The weather is sunny and 22°C"}]
                        }
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

        # Sample analyzed data
        self.sample_analyzed = [
            {
                'role': 'system',
                'type': 'system_prompt',
                'created': 0,
                'text': ["[System prompt and tools schema]"],
                'tool_requests': [],
                'tool_responses': [],
                'token_count': 3000,
                'context_tokens': 0,
                'input_tokens': 3000,
                'output_tokens': 0,
                'details': "System prompt and tools schema (3000 tokens)"
            },
            {
                'role': 'user',
                'type': 'user_input',
                'created': 1650000000,
                'text': ["What is the weather?"],
                'tool_requests': [],
                'tool_responses': [],
                'token_count': 25,
                'context_tokens': 3000,
                'input_tokens': 25,
                'output_tokens': 0,
                'details': 'What is the weather?'
            },
            {
                'role': 'assistant',
                'type': 'agent_output',
                'created': 1650000010,
                'text': ["I will check the weather for you."],
                'tool_requests': [{"id": "tool-1", "name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}],
                'tool_responses': [],
                'token_count': 50,
                'context_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 50,
                'details': 'I will check the weather for you.'
            },
            {
                'role': 'user',
                'type': 'tool_call',
                'created': 1650000020,
                'text': [],
                'tool_requests': [],
                'tool_responses': [{"id": "tool-1", "content": [{"type": "text", "text": "The weather is sunny and 22°C"}]}],
                'token_count': 30,
                'context_tokens': 3075,
                'input_tokens': 30,
                'output_tokens': 0,
                'details': 'get_weather(location:\'San Francisco\', unit:\'celsius\')'
            },
            {
                'role': 'assistant',
                'type': 'agent_output',
                'created': 1650000030,
                'text': ["The weather in San Francisco is currently sunny with a temperature of 22°C."],
                'tool_requests': [],
                'tool_responses': [],
                'token_count': 100,
                'context_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 100,
                'details': 'The weather in San Francisco is currently sunny with a temperature of 22°C.'
            }
        ]

    @patch('tiktoken.get_encoding')
    def test_tokenizer_get_tokenizer(self, mock_get_encoding):
        # Setup mock
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # 3 tokens
        mock_get_encoding.return_value = mock_encoding

        # Test the tokenizer
        tokenizer = Tokenizer('tiktoken')
        result = tokenizer.tokenize("test text")

        # Verify
        self.assertEqual(result, 3)
        mock_get_encoding.assert_called_once_with("cl100k_base")

    def test_extract_text_from_content(self):
        # Test extracting text from mixed content
        result = self.processor.extract_text_from_content(self.sample_content)

        self.assertEqual(result, ["Hello world"])

    def test_extract_tool_data(self):
        # Test tool request extraction
        tool_requests = self.processor.extract_tool_data(self.sample_content, "toolRequest")
        self.assertEqual(len(tool_requests), 1)
        self.assertEqual(tool_requests[0]["id"], "tool-1")
        self.assertEqual(tool_requests[0]["name"], "get_weather")
        self.assertEqual(tool_requests[0]["arguments"], {"location": "San Francisco", "unit": "celsius"})

        # Test tool response extraction
        tool_responses = self.processor.extract_tool_data(self.sample_content, "toolResponse")
        self.assertEqual(len(tool_responses), 1)
        self.assertEqual(tool_responses[0]["id"], "tool-1")
        self.assertEqual(tool_responses[0]["content"], [{"type": "text", "text": "The weather is sunny and 22°C"}])

    def test_count_tool_tokens(self):
        # Test counting tokens for tool requests
        tool_requests = self.processor.extract_tool_data(self.sample_content, "toolRequest")
        request_tokens = self.tokenizer.count_tool_tokens(tool_requests, is_request=True)

        # The format is "{id}:{name}:{arguments_json}" which should have multiple tokens
        self.assertGreater(request_tokens, 0)

        # Test counting tokens for tool responses
        tool_responses = self.processor.extract_tool_data(self.sample_content, "toolResponse")
        response_tokens = self.tokenizer.count_tool_tokens(tool_responses, is_request=False)

        # The format is "{id}:{response_text}" which should have multiple tokens
        self.assertGreater(response_tokens, 0)

    def test_count_tools_for_schema(self):
        # Test counting tokens for tool schema
        schema_tokens = self.tokenizer.count_tools_for_schema(self.sample_tools)

        # The schema should have tokens for name, description, properties, etc.
        self.assertGreater(schema_tokens, 0)

        # Test with empty tools list
        empty_schema_tokens = self.tokenizer.count_tools_for_schema([])
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
        enum_schema_tokens = self.tokenizer.count_tools_for_schema(tools_with_more_enums)
        self.assertGreater(enum_schema_tokens, schema_tokens,
                        "Tools with more enum values should have more tokens")

    def test_analyze_logs(self):
        # Use self.sample_logs instead of creating new mock logs
        analyzed = self.analyzer.analyze_logs(self.sample_logs)

        # Check that we have the correct number of messages
        # (4 messages in sample_logs + 1 system message added by analyze_logs)
        self.assertEqual(len(analyzed), 5)

        # Check that the first message is the system message
        self.assertEqual(analyzed[0]['role'], 'system')
        self.assertEqual(analyzed[0]['type'], 'system_prompt')

        # Check that the second message is the first user message
        self.assertEqual(analyzed[1]['role'], 'user')
        self.assertEqual(analyzed[1]['type'], 'user_input')

        # Check that context tokens are calculated correctly
        # First user message has context of system prompt
        self.assertGreater(analyzed[1]['context_tokens'], 0)

        # Check that input/output tokens are assigned correctly
        self.assertEqual(analyzed[0]['input_tokens'], analyzed[0]['token_count'])  # System message
        self.assertEqual(analyzed[1]['input_tokens'], analyzed[1]['token_count'])  # User message
        self.assertEqual(analyzed[2]['output_tokens'], analyzed[2]['token_count'])  # Assistant message

    def test_print_analysis(self):
        # Create formatter
        formatter = AnalysisFormatter()

        # Capture stdout to verify output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            formatter.print_analysis(self.sample_analyzed)

        output = captured_output.getvalue()

        # Check that key sections are in the output
        self.assertIn("=== Session Details ===", output)
        self.assertIn("=== Session Metrics ===", output)
        self.assertIn("=== Token Usage Distribution ===", output)

        # Check that metrics are included
        self.assertIn("Total interactions", output)
        self.assertIn("Total tokens", output)

    @patch('builtins.open')
    def test_read_jsonl_file(self, mock_open):
        # Mock file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value.__iter__.return_value = [
            '{"key": "value1"}\n',
            '\n',  # Empty line should be skipped
            '{"key": "value2"}\n'
        ]
        mock_open.return_value = mock_file

        # Test reading JSONL file
        result = self.processor.read_jsonl_file('test.jsonl')

        # Should have 2 objects (skipping empty line)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['key'], 'value1')
        self.assertEqual(result[1]['key'], 'value2')

    def test_text_extraction(self):
        # Example messages from the session log
        user_message = {"role":"user","created":1745363226,"content":[{"type":"text","text":"Can you find the models database in notion?"}]}
        assistant_message = {"role":"assistant","created":1745363236,"content":[{"type":"text","text":"I'll help you search for a database in Notion that's related to models. Let me use the Notion search functionality to look for this."},{"type":"toolRequest","id":"tooluse_fLqIgCUSTfi1wm5N8l_wqB","toolCall":{"status":"success","value":{"name":"notion__API-post-search","arguments":{"filter":{"property":"object","value":"database"},"query":"models"}}}}]}
        tool_response_message = {"role":"user","created":1745363236,"content":[{"type":"toolResponse","id":"tooluse_fLqIgCUSTfi1wm5N8l_wqB","toolResult":{"status":"success","value":[{"type":"text","text":"{\"object\":\"list\",\"results\":[{\"object\":\"database\",\"id\":\"1d9f8a6e-5b6d-8048-86ce-d0136ff03c7e\",\"cover\":null,\"icon\":null,\"created_time\":\"2025-04-18T13:47:00.000Z\",..."}]}}]}

        user_text = self.processor.extract_text_from_content(user_message['content'])
        assistant_text = self.processor.extract_text_from_content(assistant_message['content'])

        # For tool response, we need to extract from the value array
        tool_response_content = tool_response_message['content'][0]['toolResult']['value']
        tool_response_text = self.processor.extract_text_from_content(tool_response_content)

        # Expected results
        expected_user_text = ["Can you find the models database in notion?"]
        expected_assistant_text = ["I'll help you search for a database in Notion that's related to models. Let me use the Notion search functionality to look for this."]
        expected_tool_response_text = ["{\"object\":\"list\",\"results\":[{\"object\":\"database\",\"id\":\"1d9f8a6e-5b6d-8048-86ce-d0136ff03c7e\",\"cover\":null,\"icon\":null,\"created_time\":\"2025-04-18T13:47:00.000Z\",..."]

        # Validate
        self.assertEqual(user_text, expected_user_text)
        self.assertEqual(assistant_text, expected_assistant_text)
        self.assertEqual(tool_response_text, expected_tool_response_text)

    def test_tokenizer_with_invalid_name(self):
        """Test that an invalid tokenizer name raises NotImplementedError"""
        with self.assertRaises(NotImplementedError):
            tokenizer = Tokenizer("invalid_tokenizer")

    def test_count_tool_tokens_with_empty_data(self):
        """Test counting tokens with empty tool data"""
        tokens = self.tokenizer.count_tool_tokens([], is_request=True)
        self.assertEqual(tokens, 0)

        tokens = self.tokenizer.count_tool_tokens([], is_request=False)
        self.assertEqual(tokens, 0)

    def test_count_tool_tokens_with_complex_structure(self):
        """Test counting tokens with complex nested tool data"""
        complex_tool = [{
            "id": "complex-tool",
            "name": "complex_function",
            "arguments": {
                "nested": {
                    "deeply": {
                        "nested": "value"
                    }
                },
                "array": [1, 2, 3, 4, 5]
            }
        }]

        tokens = self.tokenizer.count_tool_tokens(complex_tool, is_request=True)
        self.assertGreater(tokens, 0)

    def test_extract_tool_data_with_invalid_content(self):
        """Test extracting tool data from invalid content"""
        # Test with non-list content
        result = self.processor.extract_tool_data("not a list", "toolRequest")
        self.assertEqual(result, [])

        # Test with empty content
        result = self.processor.extract_tool_data([], "toolRequest")
        self.assertEqual(result, [])

        # Test with content missing required fields
        result = self.processor.extract_tool_data([{"type": "toolRequest"}], "toolRequest")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "")
        self.assertEqual(result[0]["name"], "")
        self.assertEqual(result[0]["arguments"], {})

    def test_format_tool_call_details_with_complex_args(self):
        """Test formatting tool call details with complex arguments"""
        tool_requests = [{
            "id": "tool-1",
            "name": "complex_function",
            "arguments": {
                "param1": "value1",
                "param2": "value2",
                "param3": "value3",
                "param4": "value4"
            }
        }]

        tool_responses = [{
            "id": "tool-1",
            "content": [{"type": "text", "text": "Response"}]
        }]

        result = self.processor.format_tool_call_details(tool_requests, tool_responses)

        # Should truncate and show +2 more
        self.assertIn("complex_function", result)
        self.assertIn("+2 more", result)

    def test_extract_message_data_with_different_roles(self):
        """Test extracting message data with different roles"""
        # Test with user message
        user_msg = {
            "role": "user",
            "created": 1650000000,
            "content": [{"type": "text", "text": "User message"}]
        }
        result = self.analyzer.extract_message_data(user_msg)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["type"], "user_input")

        # Test with assistant message
        assistant_msg = {
            "role": "assistant",
            "created": 1650000010,
            "content": [{"type": "text", "text": "Assistant message"}]
        }
        result = self.analyzer.extract_message_data(assistant_msg)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["type"], "agent_output")

        # Test with unknown role
        unknown_msg = {
            "role": "unknown",
            "created": 1650000020,
            "content": [{"type": "text", "text": "Unknown message"}]
        }
        result = self.analyzer.extract_message_data(unknown_msg)
        self.assertEqual(result["role"], "unknown")
        self.assertEqual(result["type"], "unknown")

    def test_calculate_context_sizes_with_multiple_messages(self):
        """Test calculating context sizes with multiple messages"""
        messages = [
            {
                "role": "system",
                "type": "system_prompt",
                "token_count": 100
            },
            {
                "role": "user",
                "type": "user_input",
                "token_count": 50
            },
            {
                "role": "assistant",
                "type": "agent_output",
                "token_count": 75
            },
            {
                "role": "user",
                "type": "user_input",
                "token_count": 25
            }
        ]

        result = self.analyzer.calculate_context_sizes(messages)

        # System message should have no context
        self.assertEqual(result[0]["context_tokens"], 0)

        # First user message should have system message as context
        self.assertEqual(result[1]["context_tokens"], 100)

        # Second user message should have all previous messages as context
        self.assertEqual(result[3]["context_tokens"], 100 + 50 + 75)

def test_outlier_detection_with_iqr(self):
    """Test outlier detection using the IQR method"""
    import pandas as pd

    # Create a dataset with clear outliers
    data = [
        {"created": 1650000000, "input_tokens": 50, "output_tokens": 0},
        {"created": 1650000010, "input_tokens": 0, "output_tokens": 75},
        {"created": 1650000020, "input_tokens": 60, "output_tokens": 0},
        {"created": 1650000030, "input_tokens": 0, "output_tokens": 80},
        {"created": 1650000040, "input_tokens": 55, "output_tokens": 0},
        {"created": 1650000050, "input_tokens": 0, "output_tokens": 70},
        {"created": 1650000060, "input_tokens": 500, "output_tokens": 0}  # Outlier
    ]

    formatter = AnalysisFormatter()
    df = formatter.prepare_dataframe(data)

    # Verify that the outlier is flagged
    self.assertEqual(df.iloc[6]["flag"], "<--OUTLIER(I+O)")

    # Verify that non-outliers are not flagged
    for i in range(6):
        self.assertEqual(df.iloc[i]["flag"], "")

    def test_outlier_detection_with_small_dataset(self):
        """Test outlier detection with a dataset too small for IQR"""
        import pandas as pd

        # Create a small dataset (less than 4 points)
        data = [
            {"created": 1650000000, "input_tokens": 50, "output_tokens": 0},
            {"created": 1650000010, "input_tokens": 0, "output_tokens": 75},
            {"created": 1650000020, "input_tokens": 1000, "output_tokens": 0}  # Would be an outlier in larger dataset
        ]

        formatter = AnalysisFormatter()
        df = formatter.prepare_dataframe(data)

        # Verify that no flags are set for small datasets
        for i in range(len(df)):
            self.assertEqual(df.iloc[i]["flag"], "")

    def test_outlier_detection_with_small_dataset(self):
        """Test outlier detection with a dataset too small for IQR"""
        import pandas as pd

        # Create a small dataset (less than 4 points)
        data = [
            {"input_tokens": 50, "output_tokens": 0},
            {"input_tokens": 0, "output_tokens": 75},
            {"input_tokens": 1000, "output_tokens": 0}  # Would be an outlier in larger dataset
        ]

        formatter = AnalysisFormatter()
        df = formatter.prepare_dataframe(data)

        # Verify that no flags are set for small datasets
        for i in range(len(df)):
            self.assertEqual(df.iloc[i]["flag"], "")


    def test_format_session_details_with_different_col_widths(self):
        """Test formatting session details with different column widths"""
        data = [
            {
                "role": "user",
                "type": "user_input",
                "created": 1650000000,
                "input_tokens": 50,
                "output_tokens": 0,
                "context_tokens": 0,
                "details": "A very long details string that should be truncated"
            }
        ]

        formatter = AnalysisFormatter()
        df = formatter.prepare_dataframe(data)

        # Test with default column width
        result_default = formatter.format_session_details(df)

        # Test with smaller column width
        result_small = formatter.format_session_details(df, col_width=20)

        # Test with no truncation
        result_no_truncation = formatter.format_session_details(df, col_width=0)

        # The default result should be longer than the small result
        self.assertGreater(len(result_default), len(result_small))

        # The no truncation result should show the full details
        self.assertIn("A very long details string that should be truncated", result_no_truncation)

    @patch('builtins.open')
    def test_read_jsonl_file_with_invalid_json(self, mock_open):
        """Test reading JSONL file with invalid JSON"""
        # Mock file with invalid JSON
        mock_file = MagicMock()
        mock_file.__enter__.return_value.__iter__.return_value = [
            '{"key": "value"}\n',
            'not valid json\n',
            '{"key": "value2"}\n'
        ]
        mock_open.return_value = mock_file

        # Should raise JSONDecodeError
        with self.assertRaises(json.JSONDecodeError):
            self.processor.read_jsonl_file('test.jsonl')

    @patch('os.path.exists')
    def test_main_with_nonexistent_file(self, mock_exists):
        """Test main function with nonexistent file"""
        mock_exists.return_value = False

        # Mock args to use a nonexistent file
        with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_args = MagicMock()
            mock_args.input_file = 'nonexistent.jsonl'
            mock_args.tokenizer = "tiktoken"
            mock_args.col_width = 100
            mock_parse_args.return_value = mock_args

            # Should raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                token_analysis.main()
                
if __name__ == '__main__':
    unittest.main()