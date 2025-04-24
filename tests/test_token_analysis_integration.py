import unittest
import os
import sys
from unittest.mock import patch
import io
from contextlib import redirect_stdout

# Add parent directory to path so we can import token_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main function from token_analysis.py
from token_analysis import main, analyze_logs

class TestTokenAnalysisIntegration(unittest.TestCase):
    def setUp(self):
        # Path to the test data file
        self.test_file = os.path.join('test_data', 'sample_session.jsonl')
        
        # Ensure the test file exists
        if not os.path.exists(self.test_file):
            self.skipTest(f"Test file {self.test_file} not found")
    
    @patch('token_analysis.tiktoken')
    def test_analyze_logs_with_real_file(self, mock_tiktoken):
        # Setup mock tokenizer to return predictable values
        mock_encoding = mock_tiktoken.get_encoding.return_value
        mock_encoding.encode.return_value = [1, 2, 3]  # Always return 3 tokens
        
        # Read and analyze the test file
        from token_analysis import read_jsonl_file
        session_logs = read_jsonl_file(self.test_file)
        
        # Analyze the logs
        analyzed = analyze_logs(session_logs, tokenizer_name='tiktoken')
        
        # Basic validation
        self.assertIsNotNone(analyzed)
        self.assertGreater(len(analyzed), 0)
        
        # Check that we have the expected message types
        message_types = [msg['type'] for msg in analyzed]
        self.assertIn('user_input', message_types)
        self.assertIn('agent_output', message_types)
        
        # Check that token counts are present
        for msg in analyzed:
            self.assertIn('input_tokens', msg)
            self.assertIn('output_tokens', msg)
            self.assertIn('context_tokens', msg)
            
        # Verify that the first user message has system overhead added
        first_user_msg = next((msg for msg in analyzed if msg['role'] == 'user'), None)
        self.assertIsNotNone(first_user_msg)
        self.assertGreater(first_user_msg['input_tokens'], 3000)  # SYSTEM_PROMPT_OVERHEAD
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('token_analysis.tiktoken')
    def test_main_function(self, mock_tiktoken, mock_parse_args):
        # Setup mock tokenizer
        mock_encoding = mock_tiktoken.get_encoding.return_value
        mock_encoding.encode.return_value = [1, 2, 3]  # Always return 3 tokens
        
        # Setup mock args
        mock_args = mock_parse_args.return_value
        mock_args.input_file = self.test_file
        mock_args.tokenizer = 'tiktoken'
        
        # Capture stdout to check output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            main()
        
        output = captured_output.getvalue()
        
        # Check that the output contains expected sections
        self.assertIn("=== Session Details ===", output)
        self.assertIn("=== Session Summary ===", output)
        self.assertIn("=== Token Usage Distribution ===", output)
        
        # Check that the analysis completed
        self.assertIn("Analysis complete!", output)

if __name__ == '__main__':
    unittest.main()
