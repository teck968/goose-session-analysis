import json
import pandas as pd
import argparse
from datetime import datetime

def get_tokenizer(tokenizer_name):
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

def extract_by_criteria(obj, match_fn, value_fn, join_str=" "):
    """Extract values from nested data structures based on matching criteria"""
    results = []

    def walk(o):
        if isinstance(o, list):
            for item in o:
                walk(item)
        elif isinstance(o, dict):
            if match_fn(o):
                results.append(value_fn(o))
            else:
                for v in o.values():
                    walk(v)
        elif match_fn(o):
            results.append(value_fn(o))

    walk(obj)
    return join_str.join(str(r) for r in results if r is not None)

def count_tool_tokens(tokenizer, tool_data, is_request=True):
    """Count tokens for tool requests or responses based on Goose's logic"""
    tokens = 0

    for tool in tool_data:
        if is_request:
            # Format matches token_counter.rs implementation
            tool_text = f"{tool['id']}:{tool['name']}:{json.dumps(tool['arguments'])}"
            tokens += tokenizer(tool_text)
        else:
            # Extract text from tool response content, matching Rust implementation
            text_items = []
            for item in tool.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    text_items.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_items.append(item)

            # Join with newlines to match Rust's texts.join("\n")
            response_text = "\n".join(text_items)

            # Format matches message.rs implementation: just id + text
            tokens += tokenizer(f"{tool['id']}:{response_text}")

    return tokens

def count_tools_for_schema(tokenizer, tools):
    """Count tokens for tools schema following Goose's count_tokens_for_tools logic"""
    # TODO[ID:validate_toolschema_present]: Validate that this function works as expected on goose session logs. I suspect it doesn't because the data is not present in the logs.
    # Constants from Goose's token_counter.rs
    FUNC_INIT = 7      # Function initialization
    PROP_INIT = 3      # Properties initialization
    PROP_KEY = 3       # Each property key
    ENUM_INIT = -3     # Enum list start adjustment
    ENUM_ITEM = 3      # Each enum item
    FUNC_END = 12      # Function ending
    
    if not tools:
        return 0
        
    count = 0
    for tool in tools:
        count += FUNC_INIT
        # Name and description - strip trailing periods as in Goose
        name = tool.get('name', '')
        description = tool.get('description', '').rstrip('.')
        count += tokenizer(f"{name}:{description}")
        
        # Properties
        properties = tool.get('input_schema', {}).get('properties', {})
        if properties:
            count += PROP_INIT
            for key, value in properties.items():
                count += PROP_KEY
                p_type = value.get('type', '')
                p_desc = value.get('description', '').rstrip('.')
                count += tokenizer(f"{key}:{p_type}:{p_desc}")
                
                # Enum values - exactly matching Goose's implementation
                enum_values = value.get('enum', [])
                if enum_values:
                    count += ENUM_INIT  # Adjustment for enum list start
                    for item in enum_values:
                        count += ENUM_ITEM  # Fixed overhead per enum item
                        count += tokenizer(str(item))  # Tokens for the enum value itself
    
    count += FUNC_END
    return count

def extract_tool_data(content, tool_type):
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

def format_tool_call_details(tool_requests, tool_responses):
    """Format tool call details to show function name and parameter values"""
    if not tool_responses:
        return ""

    # Match tool responses with their requests
    details = []
    for response in tool_responses:
        response_id = response.get("id", "")
        # Find matching request
        matching_request = next((req for req in tool_requests if req.get("id") == response_id), None)

        if matching_request:
            # Get function name
            func_name = matching_request.get("name", "unknown")

            # Format arguments compactly with values
            args = matching_request.get("arguments", {})
            if isinstance(args, dict):
                # Format as key:value pairs, but limit to first 2 args for brevity
                arg_items = list(args.items())
                if len(arg_items) > 2:
                    # For complex values, just show a brief representation
                    arg_str = ", ".join(f"{k}:{repr(v)[:50]}" for k, v in arg_items[:2])
                    arg_str += f", +{len(arg_items)-2} more"
                else:
                    arg_str = ", ".join(f"{k}:{repr(v)[:50]}" for k, v in arg_items)

                # Replace long strings with truncated versions
                arg_str = arg_str.replace("'", "")
            else:
                arg_str = str(args)

            details.append(f"{func_name}({arg_str})")
        else:
            details.append(response_id)

    return ", ".join(details)

def analyze_logs(session_logs, tokenizer_name):
    """Analyze token usage in a Goose session log"""
    tokenizer = get_tokenizer(tokenizer_name)
    summary = session_logs[0]
    messages = [entry for entry in session_logs[1:] if 'role' in entry and 'content' in entry]

    # Constants from Goose codebase
    TOKENS_PER_MESSAGE = 4
    ASSISTANT_REPLY_TOKENS = 3
    SYSTEM_PROMPT_OVERHEAD = 3000

    # Extract tools and calculate schema tokens
    tools = summary.get('tools', [])
    tools_schema_tokens = count_tools_for_schema(tokenizer, tools)  # TODO[ID:validate_toolschema_present]: Validate that this function works as expected on goose session logs. I suspect it doesn't because the data is not present in the logs.

    # PHASE 1: Extract text and metadata from all messages
    processed_messages = []

    # Add system prompt as the first message
    processed_messages.append({
        'role': 'system',
        'type': 'system_prompt',
        'created': 0,
        'text': f"[System prompt and tools schema]",
        'tool_requests': [],
        'tool_responses': [],
        'token_count': SYSTEM_PROMPT_OVERHEAD + tools_schema_tokens,
        'details': f"System prompt and tools schema ({SYSTEM_PROMPT_OVERHEAD + tools_schema_tokens} tokens)"
    })

    # PHASE 1: Process each actual message
    for msg in messages:
        content = msg['content']
        role = msg.get('role', '')
        created = msg.get('created', 0)

        # Extract text and tool data
        all_text = extract_by_criteria(
            content,
            match_fn=lambda x: isinstance(x, str) or (isinstance(x, dict) and x.get("type") == "text"),
            value_fn=lambda x: x if isinstance(x, str) else x.get("text", "")
        )

        tool_requests = extract_tool_data(content, "toolRequest")  # TODO[ID:validate_extract_toolRequest_data]: Validate that the results match Goose's implementation
        tool_responses = extract_tool_data(content, "toolResponse")  # TODO[ID:validate_extract_toolResponse_data]: Validate that the results match Goose's implementation

        # Determine message type
        msg_type = "unknown"
        if role == 'user':
            msg_type = "tool_call" if tool_responses else "user_input"
        elif role == 'assistant':
            msg_type = "agent_output"

        # Create display details
        if msg_type in ("user_input", "agent_output"):
            details = all_text
        elif msg_type == "tool_call":
            # Find the previous assistant message to get the tool requests
            prev_assistant = next((m for m in reversed(processed_messages)
                                  if m['role'] == 'assistant'), None)

            prev_tool_requests = prev_assistant['tool_requests'] if prev_assistant else []
            details = format_tool_call_details(prev_tool_requests, tool_responses)
        else:
            details = ""

        processed_messages.append({
            'role': role,
            'type': msg_type,
            'created': created,
            'text': all_text,
            'tool_requests': tool_requests,
            'tool_responses': tool_responses,
            'details': details
        })  

    # PHASE 2: Calculate token counts for each message
    for msg in processed_messages:
        # Skip if token count is already calculated (system prompt)
        if 'token_count' in msg:
            continue

        # Calculate message overhead
        message_overhead = TOKENS_PER_MESSAGE
        if msg['role'] == 'assistant':
            message_overhead += ASSISTANT_REPLY_TOKENS

        # Count tokens for content
        text_tokens = tokenizer(msg['text']) if msg['text'] else 0
        tool_request_tokens = count_tool_tokens(tokenizer, msg['tool_requests'], is_request=True)  # TODO[ID:validate_tool_request_tokens]: Validate that the results match Goose's implementation
        tool_response_tokens = count_tool_tokens(tokenizer, msg['tool_responses'], is_request=False)  # TODO[ID:validate_tool_response_tokens]: Validate that the results match Goose's implementation

        # Total tokens for this message
        msg['token_count'] = text_tokens + tool_request_tokens + tool_response_tokens + message_overhead  # TODO[ID:validate_total_tokens]: Validate that the results match Goose's implementation

    # PHASE 3: Calculate context sizes for each message
    for i, msg in enumerate(processed_messages):
        # Initialize context_tokens, input_tokens, and output_tokens if not present
        msg['context_tokens'] = 0
        msg['input_tokens'] = 0
        msg['output_tokens'] = 0

        # Only calculate context tokens for user messages
        if msg['role'] == 'user' and i > 0:
            # Context is the sum of all previous messages' token counts
            msg['context_tokens'] = sum(m['token_count'] for m in processed_messages[:i])

        # Set input or output tokens based on role
        if msg['role'] in ['system', 'user']:
            msg['input_tokens'] = msg['token_count']
        elif msg['role'] == 'assistant':
            msg['output_tokens'] = msg['token_count']

    return processed_messages


def print_analysis(token_logs, col_width=100):
    """Print a formatted analysis of token usage"""
    df = pd.DataFrame(token_logs)
    
    # Add calculated columns
    df['datetime'] = pd.to_datetime(df['created'], unit='s', errors='coerce')
    df['total_io_tokens'] = df['input_tokens'] + df['output_tokens']
    
    # Mark outliers (> 3 std dev from mean)
    threshold = df['total_io_tokens'].mean() + 3 * df['total_io_tokens'].std()
    df['flag'] = df['total_io_tokens'].apply(lambda x: "<--OUTLIER" if x > threshold else "")

    # Set the details column width if not zero
    if col_width > 0:
        df['details'] = df['details'].astype(str).str.ljust(col_width)
        max_colwidth = col_width
    else:
        # No truncation when col_width is 0
        max_colwidth = None

    # Print session details table
    print("\n=== Session Details ===")

    print(df[['datetime', 'created', 'type', 'context_tokens', 'input_tokens', 'output_tokens', 'flag', 'details']].to_string(index=False, max_colwidth=col_width))
    
    # Calculate system overhead
    system_overhead = 0
    if not df.empty:
        first_user = df[df['role'] == 'user'].iloc[0] if not df[df['role'] == 'user'].empty else None
        if first_user is not None:
            # Extract system overhead by subtracting message content tokens
            user_msg_tokens = first_user['total_io_tokens'] - first_user['input_tokens']
            system_overhead = first_user['input_tokens'] - user_msg_tokens
            system_overhead = max(0, system_overhead)
    
    # Calculate total tokens
    total_tokens = system_overhead + df['input_tokens'].sum() + df['output_tokens'].sum()
    
    # Create metrics table
    metrics = [
        ("Total interactions", len(df)),
        ("System prompt & tools overhead", f"{system_overhead:,}"),
        ("Total context tokens", f"{df['context_tokens'].sum():,}"),
        ("Total user input tokens", f"{df[df['type'] == 'user_input']['input_tokens'].sum() - system_overhead:,}"),
        ("Total tool input tokens", f"{df[df['type'] == 'tool_call']['input_tokens'].sum():,}"),
        ("Total agent output tokens", f"{df['output_tokens'].sum():,}"),
        ("Total tokens (including overhead)", f"{total_tokens:,}"),
        ("Average tokens per interaction", f"{df['total_io_tokens'].mean():.1f}")
    ]

    print("\n=== Session Summary ===")
    print(pd.DataFrame(metrics, columns=["Metric", "Value"]).to_string(index=False))

    # Print top token users
    print("\n=== Token Usage Distribution ===")
    print("Top 10 most token-intensive interactions:")
    top_columns = ['datetime', 'created', 'type', 'total_io_tokens', 'flag']
    print(df.nlargest(10, 'total_io_tokens')[top_columns].to_string(index=False))

def read_jsonl_file(filepath):
    """Read a JSONL file and return a list of parsed JSON objects"""
    session_logs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                session_logs.append(json.loads(line))
    return session_logs

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
    session_logs = read_jsonl_file(args.input_file)
    token_logs = analyze_logs(session_logs, tokenizer_name=args.tokenizer)
    print_analysis(token_logs, col_width=args.col_width)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

