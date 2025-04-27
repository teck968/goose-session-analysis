import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout


# Add parent directory to path so we can import token_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main function from token_analysis.py
# from token_analysis import main, analyze_logs
import token_analysis

class TestTokenAnalysisIntegration(unittest.TestCase):
    def setUp(self):
        # Path to the test data file
        self.test_file = os.path.join('tests', 'test_data', 'sample_session.jsonl')
        
        # Ensure the test file exists
        if not os.path.exists(self.test_file):
            self.skipTest(f"Test file {self.test_file} not found")
    
    @patch('tiktoken.get_encoding')
    def test_analyze_logs_with_real_file(self, mock_get_encoding):
        # Setup mock tokenizer
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda text: text.split()
        mock_get_encoding.return_value = mock_enc

        # Read the test file
        session_logs = token_analysis.read_jsonl_file(self.test_file)

        # Analyze logs
        analyzed = token_analysis.analyze_logs(session_logs, "tiktoken")

        # Basic validation
        self.assertGreater(len(analyzed), 0)

        # First message should be system prompt
        first_msg = analyzed[0]
        self.assertEqual(first_msg['role'], 'system')
        self.assertEqual(first_msg['type'], 'system_prompt')

        # System prompt should have the overhead tokens
        self.assertGreaterEqual(first_msg['input_tokens'], 3000)  # SYSTEM_PROMPT_OVERHEAD

        # First user message should be after system prompt
        first_user_msg = next((msg for msg in analyzed if msg['role'] == 'user'), None)
        self.assertIsNotNone(first_user_msg)

        # Check that we have some context tokens in user messages (after the first)
        user_msgs = [msg for msg in analyzed if msg['role'] == 'user']
        if len(user_msgs) > 1:
            self.assertGreater(user_msgs[1]['context_tokens'], 0)

        # Check that assistant messages have output tokens
        assistant_msgs = [msg for msg in analyzed if msg['role'] == 'assistant']
        if assistant_msgs:
            self.assertGreater(assistant_msgs[0]['output_tokens'], 0)
    
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('tiktoken.get_encoding')
    def test_main_function(self, mock_get_encoding, mock_parse_args):
        # Setup mock tokenizer
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda text: text.split()
        mock_get_encoding.return_value = mock_enc

        # Setup mock args
        mock_args = MagicMock()
        mock_args.input_file = self.test_file
        mock_args.tokenizer = "tiktoken"
        mock_args.col_width = 100
        mock_parse_args.return_value = mock_args

        # Patch print function to capture output
        with patch('builtins.print') as mock_print:
            # Run main function
            token_analysis.main()

            # Check that print was called multiple times
            self.assertGreater(mock_print.call_count, 5)

            # Check that key sections are printed
            mock_print.assert_any_call("\n=== Session Details ===")
            mock_print.assert_any_call("\n=== Session Metrics ===")
            mock_print.assert_any_call("\n=== Token Usage Distribution ===")

if __name__ == '__main__':
    unittest.main()
