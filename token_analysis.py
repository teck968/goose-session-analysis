import json
import pandas as pd
import argparse

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(enc.encode(text))
        except ImportError:
            raise ImportError("tiktoken is not installed. Please install it with 'pip install tiktoken'")
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

def extract_by_criteria(obj, match_fn, value_fn, join_str=" ", recursive=True):
    """
    Extracts values from nested data structures based on a matching function.

    Parameters:
        obj: The input data (can be dict, list, or str).
        match_fn: Function that returns True if the object matches the criteria.
        value_fn: Function that extracts the value from a matching object.
        join_str: String used to join multiple results.
        recursive: If True, searches nested structures.

    Returns:
        A string with the extracted values joined by join_str.
    """
    results = []

    def walk(o):
        # If it's a list, check each item
        if isinstance(o, list):
            for item in o:
                walk(item)
        # If it's a dict, check if it matches, else check its values
        elif isinstance(o, dict):
            if match_fn(o):
                results.append(value_fn(o))
            elif recursive:
                for v in o.values():
                    walk(v)
        # If it's a string or other type, check if it matches
        elif match_fn(o):
            results.append(value_fn(o))

    walk(obj)
    # Only include non-None results, join as string
    return join_str.join(str(r) for r in results if r is not None)

def extract_tool_data(content, tool_type):
    """
    Extract tool requests or responses from message content
    
    Parameters:
        content: The message content to search
        tool_type: Either "toolRequest" or "toolResponse"
        
    Returns:
        List of tool data dictionaries
    """
    tool_data = []
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == tool_type:
                if tool_type == "toolRequest":
                    # Extract tool call details
                    tool_call = item.get("toolCall", {})
                    tool_data.append({
                        "id": item.get("id", ""),
                        "name": tool_call.get("name", ""),
                        "arguments": tool_call.get("arguments", {})
                    })
                elif tool_type == "toolResponse":
                    # Extract tool response details
                    tool_data.append({
                        "id": item.get("id", ""),
                        "content": item.get("content", [])
                    })
    
    return tool_data

def count_tool_tokens(tokenizer, tool_data, is_request=True):
    """
    Count tokens for tool requests or responses based on Goose's logic
    
    Parameters:
        tokenizer: Function to count tokens in a string
        tool_data: List of tool data dictionaries
        is_request: True if counting tool requests, False for responses
        
    Returns:
        Total token count for the tools
    """
    tokens = 0
    
    for tool in tool_data:
        if is_request:
            # Format similar to how Goose formats tool requests in token_counter.rs
            tool_text = f"{tool['id']}:{tool['name']}:{json.dumps(tool['arguments'])}"
            tokens += tokenizer(tool_text)
        else:
            # Format similar to how Goose formats tool responses
            response_text = ""
            for content_item in tool.get("content", []):
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    response_text += content_item.get("text", "")
                elif isinstance(content_item, str):
                    response_text += content_item
            
            tokens += tokenizer(f"{tool['id']}:{response_text}")
    
    return tokens

def analyze_logs(session_logs, tokenizer_name="tiktoken"):
    tokenizer = get_tokenizer(tokenizer_name)
    summary = session_logs[0]
    messages = [entry for entry in session_logs[1:] if 'role' in entry and 'content' in entry]

    # Constants from Goose codebase
    TOKENS_PER_MESSAGE = 4  # From token_counter.rs
    ASSISTANT_REPLY_TOKENS = 3  # From token_counter.rs
    SYSTEM_PROMPT_OVERHEAD = 3000  # From context_mgmt/common.rs
    TOOLS_OVERHEAD = 5000  # From context_mgmt/common.rs

    analyzed = []
    context_messages = []
    for msg in messages:
        content = msg['content']
        role = msg.get('role', '')
        created = msg.get('created', 0)
        
        # Extract all text content
        all_text = extract_by_criteria(
            content,
            match_fn=lambda x: isinstance(x, str) or (isinstance(x, dict) and x.get("type") == "text"),
            value_fn=lambda x: x if isinstance(x, str) else x.get("text", ""),
            join_str=" "
        )
        
        # Extract tool requests and responses
        tool_requests = extract_tool_data(content, "toolRequest")
        tool_responses = extract_tool_data(content, "toolResponse")
        
        # Determine message type
        if role == 'user':
            has_tool_response = bool(tool_responses)
            msg_type = "tool_call" if has_tool_response else "user_input"
        elif role == 'assistant':
            msg_type = "agent_output"
        else:
            msg_type = "unknown"
            
        # Add message to context
        msg_text = f"{role} {all_text}"
        context_messages.append(msg_text)
        
        # Calculate token counts
        message_overhead = TOKENS_PER_MESSAGE
        if role == 'assistant':
            message_overhead += ASSISTANT_REPLY_TOKENS
            
        # Count tokens for text content
        text_tokens = tokenizer(all_text) if all_text else 0
        
        # Count tokens for tool requests and responses
        tool_request_tokens = count_tool_tokens(tokenizer, tool_requests, is_request=True)
        tool_response_tokens = count_tool_tokens(tokenizer, tool_responses, is_request=False)
        
        # Calculate total tokens for this message
        total_tokens = text_tokens + tool_request_tokens + tool_response_tokens + message_overhead
        
        # Determine context, input, and output tokens
        if role == 'user':
            # Calculate context tokens from previous messages
            if context_messages[:-1]:
                context_text = " ".join(context_messages[:-1])
                context_tokens = tokenizer(context_text)
                # Add message overhead for each previous message
                context_tokens += TOKENS_PER_MESSAGE * len(context_messages[:-1])
                # Add assistant reply tokens for assistant messages
                for i, msg in enumerate(context_messages[:-1]):
                    if msg.startswith("assistant"):
                        context_tokens += ASSISTANT_REPLY_TOKENS
            else:
                context_tokens = 0
                
            input_tokens = total_tokens
            output_tokens = 0
        elif role == 'assistant':
            context_tokens = 0
            input_tokens = 0
            output_tokens = total_tokens
        else:
            context_tokens = 0
            input_tokens = 0
            output_tokens = 0
            
        # Determine details field (truncated for display)
        if msg_type in ("user_input", "agent_output"):
            details = all_text[:75] + ("..." if len(all_text) > 75 else "")
        elif msg_type == "tool_call":
            tool_ids = [tool.get("id", "") for tool in tool_responses]
            details = ", ".join(tool_ids)
        else:
            details = ""
            
        analyzed.append({
            'role': role,
            'type': msg_type,
            'created': created,
            'content': content,
            'context_tokens': context_tokens,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'details': details,
        })
    
    # Add system prompt and tools overhead to the first message if it exists
    if analyzed and any(item['role'] == 'user' for item in analyzed):
        # Find the first user message
        for item in analyzed:
            if item['role'] == 'user':
                item['input_tokens'] += SYSTEM_PROMPT_OVERHEAD + TOOLS_OVERHEAD
                break
        
    return analyzed

def print_analysis(token_logs):
    # Convert to DataFrame 
    df = pd.DataFrame(token_logs)

    # Convert 'created' to datetime
    df['datetime'] = pd.to_datetime(df['created'], unit='s', errors='coerce')

    # Mark outliers in a new column
    df['total_io_tokens'] = df['input_tokens'] + df['output_tokens']
    threshold = df['total_io_tokens'].mean() + 3 * df['total_io_tokens'].std()
    df['flag'] = df['total_io_tokens'].apply(lambda x: "<--OUTLIER" if x > threshold else "")

    print("\n=== Session Details ===")
    COL_WIDTH = 75
    df['details'] = df['details'].astype(str).str.ljust(COL_WIDTH)
    print(df[['datetime', 'created', 'type', 'context_tokens', 'input_tokens', 'output_tokens', 'flag', 'details']].to_string(index=False, max_colwidth=COL_WIDTH))
    
    # Calculate metrics with more accurate token counts
    system_overhead = 0
    if not df.empty and 'input_tokens' in df.columns:
        # Find the first user message which should have the system overhead
        first_user = df[df['role'] == 'user'].iloc[0] if not df[df['role'] == 'user'].empty else None
        if first_user is not None:
            # Extract just the system overhead portion
            system_overhead = 8000  # SYSTEM_PROMPT_OVERHEAD + TOOLS_OVERHEAD
    
    metrics = [
        ("Total interactions", len(df)),
        ("System prompt & tools overhead", f"{system_overhead:,}"),
        ("Total context tokens", f"{df['context_tokens'].sum():,}"),
        ("Total user input tokens", f"{df[df['type'] == 'user_input']['input_tokens'].sum() - system_overhead:,}"),
        ("Total tool input tokens", f"{df[df['type'] == 'tool_call']['input_tokens'].sum():,}"),
        ("Total agent output tokens", f"{df['output_tokens'].sum():,}"),
        ("Total tokens (including overhead)", f"{df['input_tokens'].sum() + df['output_tokens'].sum():,}"),
        ("Average i/o tokens per interaction", f"{(df['input_tokens'] + df['output_tokens']).mean():.1f}")
    ]

    print("\n=== Session Summary ===")
    print(pd.DataFrame(metrics, columns=["Metric", "Value"]).to_string(index=False))

    print("\n=== Token Usage Distribution ===")
    print("Top 10 most token-intensive interactions:")
    print(df.nlargest(10, 'total_io_tokens')[['datetime', 'created', 'type', 'total_io_tokens', 'flag']].to_string(index=False))

def read_jsonl_file(filepath):
    """
    Reads a JSONL file and returns a list of parsed JSON objects.
    Skips empty lines.
    """
    session_logs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                session_logs.append(json.loads(line))
    return session_logs

def main():
    parser = argparse.ArgumentParser(description='Analyze session logs for token usage')
    parser.add_argument('input_file', help='Path to the JSON log file')
    parser.add_argument('--tokenizer', default='tiktoken', help='Tokenizer to use (tiktoken, anthropic-tokenizer)')
    args = parser.parse_args()

    # Read the JSONL file with UTF-8 encoding
    session_logs = read_jsonl_file(args.input_file)

    # Analyze the logs
    token_logs = analyze_logs(session_logs, tokenizer_name=args.tokenizer)

    # Print comprehensive analysis
    print_analysis(token_logs)

if __name__ == "__main__":
    main()

