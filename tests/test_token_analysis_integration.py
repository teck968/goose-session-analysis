import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout


# Add parent directory to path so we can import token_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main function from token_analysis.py
from token_analysis import main, analyze_logs

class TestTokenAnalysisIntegration(unittest.TestCase):
    def setUp(self):
        # Path to the test data file
        self.test_file = os.path.join('tests', 'test_data', 'sample_session.jsonl')
        
        # Ensure the test file exists
        if not os.path.exists(self.test_file):
            self.skipTest(f"Test file {self.test_file} not found")
    
    @patch('tiktoken.get_encoding')
    def test_analyze_logs_with_real_file(self, mock_get_encoding):
        # Setup mock tokenizer to return predictable values
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Always return 3 tokens
        mock_get_encoding.return_value = mock_encoding
        
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
    @patch('tiktoken.get_encoding')
    def test_main_function(self, mock_get_encoding, mock_parse_args):
        # Setup mock tokenizer
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Always return 3 tokens
        mock_get_encoding.return_value = mock_encoding

        # Setup mock args - CHANGE THIS PART
        mock_args = MagicMock()
        mock_args.input_file = self.test_file
        mock_args.tokenizer = 'tiktoken'
        # Set col_width as an integer instead of letting it be a MagicMock
        mock_args.col_width = 100  # Explicitly set to an integer
        mock_parse_args.return_value = mock_args

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
