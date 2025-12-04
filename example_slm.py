"""
Example usage of small Language Models (sLM)
"""

from language_model import LanguageModelFactory

def example_slm_simple_generation():
    """Example: Simple text generation with sLM"""
    print("=" * 60)
    print("sLM Simple Text Generation Example")
    print("=" * 60)
    
    try:
        # Create sLM model instance (using distilgpt2 as default)
        model = LanguageModelFactory.create_slm(model_name="distilgpt2")
        
        prompt = "Artificial intelligence is"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...")
        
        response = model.generate(prompt, max_tokens=50, temperature=0.7)
        print(f"\nResponse:\n{response}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install required packages: pip install transformers torch")
    except Exception as e:
        print(f"\nError: {e}")


def example_slm_chat():
    """Example: Chat conversation with sLM"""
    print("\n" + "=" * 60)
    print("sLM Chat Conversation Example")
    print("=" * 60)
    
    try:
        model = LanguageModelFactory.create_slm(model_name="distilgpt2")
        
        # Simulate a simple conversation
        messages = [
            {"role": "user", "content": "What is machine learning?"}
        ]
        
        print("\nUser: What is machine learning?")
        print("\nGenerating response...")
        
        response = model.chat(messages, max_tokens=60, temperature=0.7)
        print(f"\nAssistant: {response}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install required packages: pip install transformers torch")
    except Exception as e:
        print(f"\nError: {e}")


def example_slm_creative():
    """Example: Creative text generation with sLM"""
    print("\n" + "=" * 60)
    print("sLM Creative Text Generation Example")
    print("=" * 60)
    
    try:
        model = LanguageModelFactory.create_slm(model_name="distilgpt2")
        
        prompts = [
            "Once upon a time",
            "The future of technology",
            "In a world where"
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            response = model.generate(prompt, max_tokens=40, temperature=0.8)
            print(f"Response: {response}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install required packages: pip install transformers torch")
    except Exception as e:
        print(f"\nError: {e}")


def example_slm_comparison():
    """Example: Compare different sLM models"""
    print("\n" + "=" * 60)
    print("sLM Model Comparison Example")
    print("=" * 60)
    
    # List of small models to try
    models = ["distilgpt2", "gpt2"]
    prompt = "Language models are"
    
    for model_name in models:
        try:
            print(f"\n--- Model: {model_name} ---")
            model = LanguageModelFactory.create_slm(model_name=model_name)
            
            print(f"Prompt: {prompt}")
            response = model.generate(prompt, max_tokens=40, temperature=0.7)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")


if __name__ == "__main__":
    print("\nSmall Language Model (sLM) Examples\n")
    
    # Run examples
    example_slm_simple_generation()
    example_slm_chat()
    example_slm_creative()
    # example_slm_comparison()  # Uncomment to compare different models
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
