import json
import pandas as pd
import argparse
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Union

class Tokenizer:
    """Handles tokenization and token counting operations"""

    @staticmethod
    def get_tokenizer(tokenizer_name: str) -> Callable[[str], int]:
        """Get a tokenizer function based on the model name"""
        if tokenizer_name == "tiktoken":
            try:
                import tiktoken
                # All supported models use cl100k_base
                enc = tiktoken.get_encoding("cl100k_base")
                return lambda text: len(enc.encode(text))
            except ImportError:
                raise ImportError("tiktoken is not installed. Please install it with 'pip install tiktoken'")
        else:
            raise NotImplementedError(f"Tokenizer '{tokenizer_name}' is not implemented.")

    def __init__(self, tokenizer_name: str = "tiktoken"):
        self.tokenize = self.get_tokenizer(tokenizer_name)

    def count_tool_tokens(self, tool_data: List[Dict[str, Any]], is_request: bool = True) -> int:
        """Count tokens for tool requests or responses"""
        tokens = 0

        for tool in tool_data:
            if is_request:
                tool_text = f"{tool['id']}:{tool['name']}:{json.dumps(tool['arguments'])}"
                tokens += self.tokenize(tool_text)
            else:
                text_items = []
                for item in tool.get("content", []):
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_items.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_items.append(item)

                response_text = "\n".join(text_items)
                tokens += self.tokenize(f"{tool['id']}:{response_text}")

        return tokens

    def count_tools_for_schema(self, tools: List[Dict[str, Any]]) -> int:
        """Count tokens for tools schema"""
        # Constants from Goose's token_counter.rs
        FUNC_INIT = 7
        PROP_INIT = 3
        PROP_KEY = 3
        ENUM_INIT = -3
        ENUM_ITEM = 3
        FUNC_END = 12

        if not tools:
            return 0

        count = 0
        for tool in tools:
            count += FUNC_INIT
            name = tool.get('name', '')
            description = tool.get('description', '').rstrip('.')
            count += self.tokenize(f"{name}:{description}")

            properties = tool.get('input_schema', {}).get('properties', {})
            if properties:
                count += PROP_INIT
                for key, value in properties.items():
                    count += PROP_KEY
                    p_type = value.get('type', '')
                    p_desc = value.get('description', '').rstrip('.')
                    count += self.tokenize(f"{key}:{p_type}:{p_desc}")

                    enum_values = value.get('enum', [])
                    if enum_values:
                        count += ENUM_INIT
                        for item in enum_values:
                            count += ENUM_ITEM
                            count += self.tokenize(str(item))

        count += FUNC_END
        return count


class MessageProcessor:
    """Processes and extracts data from messages"""

    @staticmethod
    def read_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
        """Read a JSONL file and return a list of parsed JSON objects"""
        session_logs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    session_logs.append(json.loads(line))
        return session_logs

    @staticmethod
    def extract_tool_data(content: List[Dict[str, Any]], tool_type: str) -> List[Dict[str, Any]]:
        """Extract tool requests or responses from message content"""
        tool_data = []

        if not isinstance(content, list):
            return tool_data

        for item in content:
            if isinstance(item, dict) and item.get("type") == tool_type:
                if tool_type == "toolRequest":
                    tool_call = item.get("toolCall", {})
                    value = tool_call.get("value", {})
                    tool_data.append({
                        "id": item.get("id", ""),
                        "name": value.get("name", ""),
                        "arguments": value.get("arguments", {})
                    })
                elif tool_type == "toolResponse":
                    tool_data.append({
                        "id": item.get("id", ""),
                        "content": item.get("toolResult", {}).get("value", [])
                    })

        return tool_data

    @staticmethod
    def extract_text_from_content(content: List[Dict[str, Any]]) -> List[str]:
        """Extract text from message content"""
        text_items = []

        if not isinstance(content, list):
            return text_items

        for item in content:
            if isinstance(item, dict) and item.get("type", "") == "text":
                text_items.append(item.get("text", ""))

        return text_items

    @staticmethod
    def format_tool_call_details(tool_requests: List[Dict[str, Any]],
                                tool_responses: List[Dict[str, Any]]) -> str:
        """Format tool call details to show function name and parameter values"""
        if not tool_responses:
            return ""

        details = []
        for response in tool_responses:
            response_id = response.get("id", "")
            matching_request = next((req for req in tool_requests if req.get("id") == response_id), None)

            if matching_request:
                func_name = matching_request.get("name", "unknown")
                args = matching_request.get("arguments", {})

                if isinstance(args, dict):
                    arg_items = list(args.items())
                    if len(arg_items) > 2:
                        arg_str = ", ".join(f"{k}:{repr(v)[:50]}" for k, v in arg_items[:2])
                        arg_str += f", +{len(arg_items)-2} more"
                    else:
                        arg_str = ", ".join(f"{k}:{repr(v)[:50]}" for k, v in arg_items)
                else:
                    arg_str = str(args)

                details.append(f"{func_name}({arg_str})")
            else:
                details.append(response_id)

        return ", ".join(details)


class TokenAnalyzer:
    """Analyzes token usage in session logs"""

    # Constants from Goose codebase
    TOKENS_PER_MESSAGE = 4
    ASSISTANT_REPLY_TOKENS = 3
    SYSTEM_PROMPT_OVERHEAD = 3000

    def __init__(self, tokenizer_name: str = "tiktoken"):
        self.tokenizer = Tokenizer(tokenizer_name)
        self.processor = MessageProcessor()

    def extract_message_data(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text and tool data from a message"""
        role = msg.get('role', '')
        created = msg.get('created', 0)
        content = msg['content']

        all_text = self.processor.extract_text_from_content(content)
        tool_requests = self.processor.extract_tool_data(content, "toolRequest")
        tool_responses = self.processor.extract_tool_data(content, "toolResponse")

        # Determine message type
        if role == 'user':
            msg_type = "tool_call" if tool_responses else "user_input"
        elif role == 'assistant':
            msg_type = "agent_output"
        else:
            msg_type = "unknown"

        return {
            'role': role,
            'type': msg_type,
            'created': created,
            'text': all_text,
            'tool_requests': tool_requests,
            'tool_responses': tool_responses
        }

    def calculate_message_tokens(self, msg: Dict[str, Any]) -> int:
        """Calculate token count for a single message"""
        # Skip if token count is already calculated
        if 'token_count' in msg:
            return msg['token_count']

        # Calculate message overhead
        message_overhead = self.TOKENS_PER_MESSAGE
        if msg['role'] == 'assistant':
            message_overhead += self.ASSISTANT_REPLY_TOKENS

        # Count tokens for content
        text_tokens = self.tokenizer.tokenize("\n".join(msg['text'])) if msg['text'] else 0
        tool_request_tokens = self.tokenizer.count_tool_tokens(msg['tool_requests'], is_request=True)
        tool_response_tokens = self.tokenizer.count_tool_tokens(msg['tool_responses'], is_request=False)

        # Total tokens for this message
        return text_tokens + tool_request_tokens + tool_response_tokens + message_overhead

    def calculate_context_sizes(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate context sizes for each message"""
        for i, msg in enumerate(messages):
            # Initialize context_tokens, input_tokens, and output_tokens
            msg['context_tokens'] = 0
            msg['input_tokens'] = 0
            msg['output_tokens'] = 0

            # Only calculate context tokens for user messages
            if msg['role'] == 'user' and i > 0:
                # Context is the sum of all previous messages' token counts
                msg['context_tokens'] = sum(m['token_count'] for m in messages[:i])

            # Set input or output tokens based on role
            if msg['role'] in ['system', 'user']:
                msg['input_tokens'] = msg['token_count']
            elif msg['role'] == 'assistant':
                msg['output_tokens'] = msg['token_count']

        return messages

    def add_message_details(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add details field to each message"""
        for i, msg in enumerate(messages):
            if msg['type'] in ("user_input", "agent_output"):
                msg['details'] = "\n".join(msg['text'])
            elif msg['type'] == "tool_call":
                # Find the previous assistant message to get the tool requests
                prev_assistant = next((m for m in reversed(messages[:i])
                                      if m['role'] == 'assistant'), None)
                prev_tool_requests = prev_assistant['tool_requests'] if prev_assistant else []
                msg['details'] = self.processor.format_tool_call_details(prev_tool_requests, msg['tool_responses'])
            else:
                msg['details'] = ""

        return messages

    def analyze_logs(self, session_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze token usage in a Goose session log"""
        summary = session_logs[0]
        messages = [entry for entry in session_logs[1:] if 'role' in entry and 'content' in entry]

        # Extract tools and calculate schema tokens
        tools = summary.get('tools', [])
        tools_schema_tokens = self.tokenizer.count_tools_for_schema(tools)

        # PHASE 1: Extract text and metadata from all messages
        processed_messages = []

        # Add system prompt as the first message
        processed_messages.append({
            'role': 'system',
            'type': 'system_prompt',
            'created': 0,
            'text': ["[System prompt and tools schema]"],
            'tool_requests': [],
            'tool_responses': [],
            'token_count': self.SYSTEM_PROMPT_OVERHEAD + tools_schema_tokens,
            'details': f"System prompt and tools schema ({self.SYSTEM_PROMPT_OVERHEAD + tools_schema_tokens} tokens)"})

        # Process each actual message
        for msg in messages:
            processed_msg = self.extract_message_data(msg)
            processed_messages.append(processed_msg)

        # PHASE 2: Calculate token counts for each message
        for msg in processed_messages:
            if 'token_count' not in msg:
                msg['token_count'] = self.calculate_message_tokens(msg)

        # PHASE 3: Calculate context sizes and add details
        processed_messages = self.calculate_context_sizes(processed_messages)
        processed_messages = self.add_message_details(processed_messages)

        return processed_messages


class AnalysisFormatter:
    """Formats and displays analysis results"""

    @staticmethod
    def prepare_dataframe(token_logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare a DataFrame from token logs for display"""
        df = pd.DataFrame(token_logs)

        # Add calculated columns
        df['datetime'] = pd.to_datetime(df['created'], unit='s', errors='coerce')
        df['total_io_tokens'] = df['input_tokens'] + df['output_tokens']

        # Mark outliers using IQR method with reduced sensitivity
        if len(df) >= 4:  # Need at least 4 points for quartiles to be meaningful
            Q1 = df['total_io_tokens'].quantile(0.25)
            Q3 = df['total_io_tokens'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 5.0 * IQR  
            min_threshold = 3000  # Only flag values above 3,000 tokens (system prompt size)

            df['flag'] = df['total_io_tokens'].apply(
                lambda x: "<--OUTLIER(I+O)" if x > upper_bound and x > min_threshold else ""
            )
        else:
            # For very small datasets, don't try to identify outliers
            df['flag'] = ""

        return df

    @staticmethod
    def format_session_details(df: pd.DataFrame, col_width: int = 100) -> str:
        """Format session details table"""
        # Set the details column width if not zero
        if col_width > 0:
            df['details'] = df['details'].astype(str).str.ljust(col_width)
            max_colwidth = col_width
        else:
            # No truncation when col_width is 0
            max_colwidth = None

        return df[['datetime', 'created', 'type', 'context_tokens',
                  'input_tokens', 'output_tokens', 'flag', 'details']].to_string(
                      index=False, max_colwidth=col_width)

    @staticmethod
    def calculate_session_metrics(df: pd.DataFrame) -> List[tuple]:
        """Calculate session metrics"""
        return [
            ("Total interactions", len(df[df['type'] != "system_prompt"])),
            ("Total context tokens", f"{df['context_tokens'].sum():,}"),
            ("Total user input tokens", f"{df[df['type'] == 'user_input']['input_tokens'].sum():,}"),
            ("Total tool input tokens", f"{df[df['type'] == 'tool_call']['input_tokens'].sum():,}"),
            ("Total agent output tokens", f"{df['output_tokens'].sum():,}"),
            ("Total tokens", f"{df['input_tokens'].sum() + df['output_tokens'].sum() + df['context_tokens'].sum():,}"),
        ]

    @staticmethod
    def format_session_metrics(metrics: List[tuple]) -> str:
        """Format session metrics table"""
        return pd.DataFrame(metrics, columns=["Metric", "Value"]).to_string(index=False)

    @staticmethod
    def format_token_distribution(df: pd.DataFrame) -> str:
        """Format token usage distribution table"""
        top_columns = ['datetime', 'created', 'type', 'total_io_tokens', 'flag']
        return df.nlargest(10, 'total_io_tokens')[top_columns].to_string(index=False)

    def print_analysis(self, token_logs: List[Dict[str, Any]], col_width: int = 100) -> None:
        """Print a formatted analysis of token usage"""
        df = self.prepare_dataframe(token_logs)

        # Print session details table
        print("\n=== Session Details ===")
        print(self.format_session_details(df, col_width))

        # Print session metrics
        metrics = self.calculate_session_metrics(df)
        print("\n=== Session Metrics ===")
        print(self.format_session_metrics(metrics))

        # Print top token interactions
        print("\n=== Token Usage Distribution ===")
        print("Top 10 most token-intensive interactions:")
        print(self.format_token_distribution(df))


def main():
    """Main entry point for the token analysis tool"""
    parser = argparse.ArgumentParser(description='Analyze Goose session logs for token usage')
    parser.add_argument('input_file', help='Path to the session log JSONL file')
    parser.add_argument('--tokenizer', default='tiktoken',
                        choices=['tiktoken'],
                        help='Tokenizer to use for counting')
    parser.add_argument('--col-width', type=int, default=100,
                        help='Column width for the details field in output (default: 100)')
    args = parser.parse_args()

    print(f"Analyzing {args.input_file} with {args.tokenizer} tokenizer...")

    # Read session logs
    processor = MessageProcessor()
    session_logs = processor.read_jsonl_file(args.input_file)

    # Analyze token usage
    analyzer = TokenAnalyzer(tokenizer_name=args.tokenizer)
    token_logs = analyzer.analyze_logs(session_logs)

    # Format and print results
    formatter = AnalysisFormatter()
    formatter.print_analysis(token_logs, col_width=args.col_width)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()