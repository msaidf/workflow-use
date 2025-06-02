# Workflow Use

A Python package for creating and executing browser-based workflows powered by LLMs.

## Configuration

The package supports multiple LLM providers through environment configuration. Copy and configure the `.env` file:

### Supported Providers

- **OpenAI** - GPT models
- **Anthropic** - Claude models  
- **Google** - Gemini models

### Environment Configuration

1. Set your provider in `.env`:
```bash
LLM_PROVIDER=openai  # or anthropic, google
```

2. Configure API keys and models:
```bash
# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
OPENAI_PAGE_EXTRACTION_MODEL=gpt-4o-mini

# Anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_PAGE_EXTRACTION_MODEL=claude-3-haiku-20240307

# Google
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-1.5-pro
GOOGLE_PAGE_EXTRACTION_MODEL=gemini-1.5-flash
```

## Usage

All CLI commands and programmatic usage will automatically use the configured provider:

```bash
# CLI usage
uv run python cli.py create-workflow
uv run python cli.py run-workflow my_workflow.json

# Programmatic usage
from workflow_use.config import create_llm_pair
llm, page_llm = create_llm_pair()
```
