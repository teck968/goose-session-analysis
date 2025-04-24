import json
import pandas as pd
import argparse
import re
from datetime import datetime

# Extract all assistant messages and compute token counts for each

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(enc.encode(text))
        except ImportError:
            raise ImportError("tiktoken is not installed. Please install it with 'pip install tiktoken'")
    elif tokenizer_name == "anthropic-tokenizer":
        try:
            from anthropic_tokenizer import Tokenizer

            tokenizer = Tokenizer()
            return lambda text: len(tokenizer.encode(text))
        except ImportError:
            raise ImportError("anthropic-tokenizer is not installed. Please install it with 'pip install anthropic-tokenizer'")
    elif tokenizer_name == "xenova-claude-tokenizer":
        raise NotImplementedError("Xenova tokenizer is only available in JavaScript environments.")
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

def analyze_logs(session_logs, tokenizer_name="tiktoken"):
    tokenizer = get_tokenizer(tokenizer_name)
    summary = session_logs[0]
    messages = [entry for entry in session_logs[1:] if 'role' in entry and 'content' in entry]

    PER_MESSAGE_OVERHEAD = 3  # Adjust for your provider

    analyzed = []
    context_messages = []
    for msg in messages:
        content = msg['content']
        all_text = extract_by_criteria(
            content,
            match_fn=lambda x: isinstance(x, str) or (isinstance(x, dict) and x.get("type") == "text"),
            value_fn=lambda x: x if isinstance(x, str) else x.get("text", ""),
            join_str=" "
        )
        role = msg.get('role', '')
        created = msg.get('created', 0)
        content = msg.get('content', {})

        msg_text = f"{role} {all_text}"
        context_messages.append(msg_text)

        # Determine msg_type
        if role == 'user':
            has_tool_response = bool(extract_by_criteria(
                content,
                match_fn=lambda x: isinstance(x, dict) and x.get("type") == "toolResponse",
                value_fn=lambda x: True,
                join_str="",
                recursive=False
            ))
            if has_tool_response:
                msg_type = "tool_call"
            else:
                msg_type = "user_input"
        elif role == 'assistant':
            msg_type = "agent_output"
        else:
            msg_type = "unknown"

        # Determine details field
        if msg_type in ("user_input", "agent_output"):
            details = extract_by_criteria(
                content,
                match_fn=lambda x: isinstance(x, dict) and x.get("type") == "text",
                value_fn=lambda x: x.get("text", ""),
                join_str=" ",
                recursive=False
            )
        elif msg_type == "tool_call":
            details = extract_by_criteria(
                content,
                match_fn=lambda x: isinstance(x, dict) and x.get("type") == "toolResponse",
                value_fn=lambda x: x.get("id", ""),
                join_str="",
                recursive=False
            )
        else:
            details = ""

        if role == 'user':
            context_text = " ".join(context_messages[:-1])
            context_tokens = tokenizer(context_text) + PER_MESSAGE_OVERHEAD * len(context_messages[:-1]) if context_text else 0
            input_tokens = tokenizer(all_text) + PER_MESSAGE_OVERHEAD
            output_tokens = 0
        elif role == 'assistant':
            context_tokens = 0
            output_tokens = tokenizer(all_text) + PER_MESSAGE_OVERHEAD
            input_tokens = 0
        else:
            context_tokens = 0
            input_tokens = 0
            output_tokens = 0

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

    return analyzed

def print_summary_stats(token_logs):

    # Convert to DataFrame 
    df = pd.DataFrame(token_logs)

    # Convert 'created' to datetime
    df['datetime'] = pd.to_datetime(df['created'], unit='s', errors='coerce')

    # Mark outliers in a new column
    df['total_io_tokens'] = df['input_tokens'] + df['output_tokens']
    threshold = df['total_io_tokens'].quantile(0.99)
    df['flag'] = df['total_io_tokens'].apply(lambda x: "<--OUTLIER" if x > threshold else "")

    print("\n=== Session Details ===")
    COL_WIDTH = 75
    df['details'] = df['details'].astype(str).str.ljust(COL_WIDTH)
    print(df[['datetime', 'created', 'type', 'context_tokens', 'input_tokens', 'output_tokens', 'flag', 'details']].to_string(index=False, max_colwidth=COL_WIDTH))
    
    metrics = [
        ("Total interactions", len(df)),
        ("Total context tokens", f"{df['context_tokens'].sum():,}"),
        ("Total user input tokens", f"{df[df['type'] == 'user_input']['input_tokens'].sum():,}"),
        ("Total tool input tokens", f"{df[df['type'] == 'tool_call']['input_tokens'].sum():,}"),
        ("Total agent output tokens", f"{df['output_tokens'].sum():,}"),
        ("Average i/o tokens per interaction", f"{(df['input_tokens'] + df['output_tokens']).mean():.1f}")
    ]

    print("\n=== Session Summary ===")
    print(pd.DataFrame(metrics, columns=["Metric", "Value"]).to_string(index=False))

    print("\n=== Token Usage Distribution ===")
    print("Top 10 most token-intensive interactions:")
    print(df.nlargest(10, 'total_io_tokens')[['datetime', 'created', 'type', 'total_io_tokens', 'flag']].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Analyze session logs for token usage')
    parser.add_argument('input_file', help='Path to the JSON log file')
    parser.add_argument('--tokenizer', default='tiktoken', help='Tokenizer to use (tiktoken, anthropic-tokenizer)')
    args = parser.parse_args()

    # Read the JSONL file with UTF-8 encoding
    session_logs = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                session_logs.append(json.loads(line))

    # Analyze the logs
    token_logs = analyze_logs(session_logs, tokenizer_name=args.tokenizer)

    # Print comprehensive analysis
    print_summary_stats(token_logs)

if __name__ == "__main__":
    main()

