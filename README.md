# Language-Model

A Python library for working with both large language models (GPT-4.1) and small language models (sLM). This project provides a unified interface for text generation and chat capabilities using different model architectures.

## Features

- **GPT-4.1 Integration**: Use OpenAI's GPT-4 models via API
- **Small Language Models (sLM)**: Run models locally using HuggingFace transformers
- **Unified Interface**: Common API for both model types
- **Multiple Use Cases**: Text generation, chat conversations, and creative writing
- **Flexible Configuration**: Environment-based configuration and factory pattern

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RAG-NLP-EWHA/Language-Model.git
cd Language-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Requirements

- Python 3.8+
- OpenAI API key (for GPT-4 usage)
- PyTorch and Transformers (for sLM usage)

## Quick Start

### Quick Demo

Run the quickstart script to see both models in action:

```bash
python quickstart.py
```

This will demonstrate:
- sLM (small language model) running locally
- GPT-4.1 text generation (if API key is configured)

### Using GPT-4.1

```python
from language_model import LanguageModelFactory

# Create GPT-4 model instance
model = LanguageModelFactory.create_gpt4()

# Simple text generation
response = model.generate("Explain quantum computing", max_tokens=100)
print(response)

# Chat conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]
response = model.chat(messages, max_tokens=150)
print(response)
```

### Using Small Language Models (sLM)

```python
from language_model import LanguageModelFactory

# Create sLM instance (runs locally)
model = LanguageModelFactory.create_slm(model_name="distilgpt2")

# Generate text
response = model.generate("Artificial intelligence is", max_tokens=50)
print(response)

# Chat conversation
messages = [{"role": "user", "content": "Tell me about AI"}]
response = model.chat(messages, max_tokens=60)
print(response)
```

## Examples

The repository includes several example scripts:

- `quickstart.py`: Quick demo of both GPT-4 and sLM
- `example_gpt4.py`: Demonstrates GPT-4.1 usage for text generation and chat
- `example_slm.py`: Shows how to use small language models locally
- `example_comparison.py`: Compares outputs from GPT-4 and sLM

Run examples:
```bash
# Quick demo (recommended to start)
python quickstart.py

# GPT-4 examples (requires API key)
python example_gpt4.py

# sLM examples (runs locally)
python example_slm.py

# Compare both models
python example_comparison.py
```

## Project Structure

```
Language-Model/
├── language_model.py      # Core language model classes
├── quickstart.py         # Quick demo script
├── example_gpt4.py       # GPT-4 usage examples
├── example_slm.py        # sLM usage examples
├── example_comparison.py # Model comparison
├── test_implementation.py # Test suite
├── requirements.txt      # Python dependencies
├── .env.example         # Environment configuration template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Architecture

### Core Classes

1. **LanguageModel (ABC)**: Abstract base class defining the interface
2. **GPT4Model**: Implementation for OpenAI's GPT-4 models
3. **SmallLanguageModel**: Implementation for local transformer models
4. **LanguageModelFactory**: Factory for creating model instances

### Key Methods

- `generate(prompt, max_tokens, temperature)`: Generate text from a prompt
- `chat(messages, max_tokens, temperature)`: Generate chat responses

## Configuration

### Environment Variables

Create a `.env` file with the following:

```env
OPENAI_API_KEY=your_openai_api_key_here
GPT_MODEL=gpt-4-turbo-preview
```

### Model Selection

**GPT-4 Models:**
- `gpt-4-turbo-preview`: Latest GPT-4 Turbo (recommended)
- `gpt-4`: Standard GPT-4
- `gpt-4-32k`: GPT-4 with extended context

**Small Language Models:**
- `distilgpt2`: Lightweight GPT-2 variant (82M params)
- `gpt2`: Original GPT-2 (117M params)
- `gpt2-medium`: Medium GPT-2 (345M params)
- `gpt2-large`: Large GPT-2 (774M params)

## Use Cases

### GPT-4.1
- Complex reasoning tasks
- High-quality content generation
- Advanced conversational AI
- Professional applications

### Small Language Models (sLM)
- Edge computing and IoT devices
- Quick prototyping and testing
- Resource-constrained environments
- Educational purposes
- Privacy-sensitive applications (local processing)

## Performance Considerations

**GPT-4:**
- Requires internet connection
- API rate limits apply
- Usage costs per token
- Excellent quality and coherence

**sLM:**
- Runs entirely locally
- Requires GPU for optimal performance
- No API costs
- Lower quality but faster inference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenAI for GPT-4 API
- HuggingFace for transformers library
- The open-source AI community