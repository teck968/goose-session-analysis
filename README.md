# Goose Session Analysis

This project provides tools for analyzing token usage in Goose AI agent session logs. It is designed to help developers and researchers understand how tokens are consumed during Goose sessions, identify outliers, and optimize workflows.

<div align="center">
  <p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
    </a>
  </p>
</div>

## Features

- **Token Usage Analysis:**  
  Analyze Goose session logs in JSONL format to compute token counts for user input, tool calls, agent output, and system overhead.

- **Detailed Reporting:**  
  Generates formatted tables summarizing each interaction, highlights outliers, and provides aggregate metrics.

- **Tokenizer:**  
  Supports the `tiktoken` tokenizer for token counting (accurate for OpenAI models).

  *Note: Using the `tiktoken` tokenizer may not be 100% accurate for other providers.*
  

- **Test Suite:**  
  Includes a test runner for validating token analysis logic.

## Requirements

- Python 3.8+
- [pandas](https://pandas.pydata.org/)
- [tiktoken](https://github.com/openai/tiktoken) (for token counting)

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Goose Session Log Locations

Goose session logs are stored in the following local directories:

- **Unix-like systems (macOS, Linux):** `~/.local/share/goose/sessions/`
- **Windows:** `%APPDATA%\Block\goose\data\sessions\`

## Usage

### Analyze a Goose Session Log

Run the analysis script on a Goose session log file (JSONL format):

```sh
python token_analysis.py <your_session_log.jsonl>
```

- The script prints a detailed breakdown of each interaction and a summary of token usage.
- By default, it uses the `tiktoken` tokenizer.

### Run Tests

To run the included unit tests:

```sh
python run_tests.py
```

## Project Structure

- `token_analysis.py`: Main script for analyzing Goose session logs and reporting token usage.
- `run_tests.py`: Test runner for executing all unit tests in the `tests/` directory.
- `requirements.txt`: Python dependencies.
- `tests/`: Directory for unit tests.

## Example Output

The analysis script provides:

- A table of session interactions with token counts and outlier flags.
- A summary of total tokens, system overhead, and averages.
- A list of the most token-intensive interactions.

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements or bug fixes.

---

**Note:**  
This project is not affiliated with the official Goose project, but is designed to work with its session log format.