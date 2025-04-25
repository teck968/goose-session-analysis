# Goose Codebase Token Analysis

This document provides a comprehensive analysis of token consumption in the Goose codebase, focusing on how tokens are calculated, processed, and logged during user sessions.

## 1. Token Handling Components

### Core Token Processing Components

#### TokenCounter (goose/crates/goose/src/token_counter.rs)
- **Primary token counting utility** that handles all token calculations
- Uses HuggingFace tokenizers to accurately count tokens for different models
- Provides methods for counting tokens in various contexts (text, tools, chat messages)

#### Context Management Components (goose/crates/goose/src/context_mgmt/common.rs)
- Contains utilities for estimating context limits and calculating token counts
- Defines constants for token overhead calculations
- Provides functions to get token counts for messages

### Supporting Structures and Types

#### Usage (goose/crates/goose/src/providers/base.rs)
```rust
pub struct Usage {
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
    pub total_tokens: Option<i32>,
}
```

#### ProviderUsage (goose/crates/goose/src/providers/base.rs)
```rust
pub struct ProviderUsage {
    pub model: String,
    pub usage: Usage,
}
```

#### ChatTokenCounts (goose/crates/goose/src/context_mgmt/common.rs)
```rust
pub struct ChatTokenCounts {
    pub system: usize,
    pub tools: usize,
    pub messages: Vec<usize>,
}
```

## 2. Token Flow Analysis

### Token Calculation Process

1. **Initialization**:
   - `TokenCounter` is initialized with a specific tokenizer name (e.g., GPT-4o, Claude)
   - The tokenizer is loaded from embedded files or downloaded from HuggingFace

2. **Message Token Counting**:
   - When a user sends a message, `count_chat_tokens` calculates tokens for the entire conversation
   - Each message has a base overhead of 4 tokens (`tokens_per_message`)
   - System prompts have additional token overhead

3. **Tool Token Counting**:
   - Tools (functions that the AI can call) have their tokens counted separately
   - `count_tokens_for_tools` calculates tokens for tool definitions
   - Various components (name, description, parameters) contribute to the token count

4. **Context Management**:
   - Before sending to the provider, the system estimates if the total tokens will exceed the model's context limit
   - If approaching the limit, messages may be truncated or summarized

5. **Provider Response**:
   - After the provider responds, token usage is recorded in the `ProviderUsage` struct
   - This includes input_tokens, output_tokens, and total_tokens

### Key Token Flow Functions

#### Token Counting (goose/crates/goose/src/token_counter.rs)
- `count_tokens(text)`: Counts tokens for a simple text string
- `count_tokens_for_tools(tools)`: Counts tokens for tool definitions
- `count_chat_tokens(system_prompt, messages, tools)`: Counts tokens for an entire conversation
- `count_everything(system_prompt, messages, tools, resources)`: Most comprehensive counting method

#### Context Management (goose/crates/goose/src/context_mgmt/common.rs)
- `estimate_target_context_limit(provider)`: Estimates safe token limit for a model
- `get_messages_token_counts(token_counter, messages)`: Gets token counts for each message
- `get_token_counts(token_counter, messages, system_prompt, tools)`: Comprehensive token counting

#### Agent Response Generation (goose/crates/goose/src/agents/agent.rs)
- `reply(messages, session)`: Main function that handles token flow during conversation
- `truncate_context(messages)`: Handles context length exceeded errors
- `summarize_context(messages)`: Summarizes conversation to reduce token count

## 3. Token-Related Constants and Parameters

### Context Limit Constants (goose/crates/goose/src/context_mgmt/common.rs)
```rust
const ESTIMATE_FACTOR: f32 = 0.7;  // Conservative estimate factor for context limit
const SYSTEM_PROMPT_TOKEN_OVERHEAD: usize = 3_000;  // Token overhead for system prompt
const TOOLS_TOKEN_OVERHEAD: usize = 5_000;  // Token overhead for tools
```

### Tokenizer Constants (goose/crates/goose/src/token_counter.rs)
```rust
let tokens_per_message = 4;  // Base token overhead per message
```

### Tool Token Counting Constants (goose/crates/goose/src/token_counter.rs)
```rust
let func_init = 7;  // Tokens for function initialization
let prop_init = 3;  // Tokens for properties initialization
let prop_key = 3;  // Tokens for each property key
let enum_init: isize = -3;  // Tokens adjustment for enum list start
let enum_item = 3;  // Tokens for each enum item
let func_end = 12;  // Tokens for function ending
```

## 4. Token Calculation Logic

### Tool Token Calculation (goose/crates/goose/src/token_counter.rs)
The token calculation for tools follows this pattern:
1. Start with base function tokens (`func_init`)
2. Add tokens for tool name and description
3. For each property:
   - Add property initialization tokens (`prop_init`)
   - Add tokens for property name, type, and description
   - For enum properties, add tokens for each enum value
4. Add function ending tokens (`func_end`)

### Chat Token Calculation (goose/crates/goose/src/token_counter.rs)
Chat token calculation follows this pattern:
1. Count system prompt tokens (if present)
2. For each message:
   - Add base message overhead (`tokens_per_message`)
   - Count tokens in message content (text, tool requests, tool responses)
3. Add tokens for tools (if present)
4. Add 3 tokens for assistant reply priming

### Context Limit Estimation (goose/crates/goose/src/context_mgmt/common.rs)
Context limit estimation:
1. Get model's maximum context limit
2. Apply conservative estimate factor (`ESTIMATE_FACTOR = 0.7`)
3. Subtract system prompt overhead (`SYSTEM_PROMPT_TOKEN_OVERHEAD = 3,000`)
4. Subtract tools overhead (`TOOLS_TOKEN_OVERHEAD = 5,000`)

## 5. Token Usage in Provider Interactions

### Provider Interface (goose/crates/goose/src/providers/base.rs)
The `Provider` trait defines the interface for AI providers:
```rust
async fn complete(
    &self,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<(Message, ProviderUsage), ProviderError>;
```

This returns both the model's response and the token usage statistics.

### Usage Tracking (goose/crates/goose/src/agents/agent.rs)
In the `reply` method, token usage is tracked:
```rust
if let Some(session_config) = session.clone() {
    Self::update_session_metrics(session_config, &usage, messages.len()).await?;
}
```

## 6. Context Length Management

### Truncation (goose/crates/goose/src/agents/context.rs)
When context length is exceeded:
1. `truncate_context` removes oldest messages to fit within context limit
2. Adds an assistant message explaining the truncation

### Summarization (goose/crates/goose/src/agents/context.rs)
Alternative to truncation:
1. `summarize_context` summarizes conversation to reduce token count
2. Adds an assistant message explaining the summarization

## 7. Session Logging

The codebase doesn't explicitly show logging of token usage, but the `update_session_metrics` function in `Agent` suggests that token usage metrics are stored in session files.

Based on the file naming pattern seen in the directory listing (`20250422_190653.jsonl`), these appear to be JSONL files that likely contain session data including token usage.

## 8. Python Analysis Recommendations

For developing a Python script to analyze token usage from Goose session logs:

1. **Parse JSONL Session Files**:
   - Look for files with `.jsonl` extension
   - Each line is likely a JSON object representing a message or event

2. **Extract Token Usage Data**:
   - Look for `ProviderUsage` objects containing:
     - `model`: The model used
     - `usage.input_tokens`: Tokens used in the input
     - `usage.output_tokens`: Tokens used in the output
     - `usage.total_tokens`: Total tokens used

3. **Calculate Aggregates**:
   - Sum token usage across messages
   - Track token usage by model
   - Calculate average tokens per message

4. **Analyze Context Management Events**:
   - Look for truncation or summarization events
   - These indicate sessions that approached context limits

5. **Tool Usage Analysis**:
   - Track token usage related to tool calls
   - Analyze which tools consume the most tokens

## 9. Key Insights for Token Analysis

1. **Conservative Estimation**: Goose uses a conservative factor (0.7) when estimating available context, leaving room for error.

2. **Overhead Accounting**: Significant token overhead is allocated for system prompts (3,000) and tools (5,000).

3. **Tool Token Complexity**: Tool token counting is complex, with different token costs for different components.

4. **Context Management Strategies**: Both truncation and summarization are used to manage context length.

5. **Provider-Specific Tokenization**: Different models use different tokenizers, affecting token counts.
