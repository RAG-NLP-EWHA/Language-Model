"""
Example usage of GPT-4.1 model
"""

from language_model import LanguageModelFactory

def example_gpt4_simple_generation():
    """Example: Simple text generation with GPT-4"""
    print("=" * 60)
    print("GPT-4 Simple Text Generation Example")
    print("=" * 60)
    
    # Create GPT-4 model instance
    # Note: Make sure to set OPENAI_API_KEY in .env file
    try:
        model = LanguageModelFactory.create_gpt4()
        
        prompt = "Explain what a small language model (sLM) is in simple terms."
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...")
        
        response = model.generate(prompt, max_tokens=150, temperature=0.7)
        print(f"\nResponse:\n{response}")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set OPENAI_API_KEY in your .env file")
    except Exception as e:
        print(f"\nError: {e}")


def example_gpt4_chat():
    """Example: Chat conversation with GPT-4"""
    print("\n" + "=" * 60)
    print("GPT-4 Chat Conversation Example")
    print("=" * 60)
    
    try:
        model = LanguageModelFactory.create_gpt4()
        
        # Simulate a conversation
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specializing in natural language processing."},
            {"role": "user", "content": "What is the difference between GPT-4 and smaller language models?"}
        ]
        
        print("\nConversation:")
        for msg in messages:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        
        print("\nGenerating response...")
        response = model.chat(messages, max_tokens=200, temperature=0.7)
        print(f"\nAssistant: {response}")
        
        # Continue the conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "What are the advantages of using sLMs?"})
        
        print(f"\nUser: {messages[-1]['content']}")
        print("\nGenerating response...")
        
        response = model.chat(messages, max_tokens=200, temperature=0.7)
        print(f"\nAssistant: {response}")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set OPENAI_API_KEY in your .env file")
    except Exception as e:
        print(f"\nError: {e}")


def example_gpt4_comparison():
    """Example: Compare GPT-4 responses with different temperatures"""
    print("\n" + "=" * 60)
    print("GPT-4 Temperature Comparison Example")
    print("=" * 60)
    
    try:
        model = LanguageModelFactory.create_gpt4()
        
        prompt = "Write a creative opening line for a story about AI."
        print(f"\nPrompt: {prompt}")
        
        for temp in [0.3, 0.7, 1.0]:
            print(f"\n--- Temperature: {temp} ---")
            response = model.generate(prompt, max_tokens=50, temperature=temp)
            print(f"Response: {response}")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set OPENAI_API_KEY in your .env file")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    print("\nGPT-4.1 Model Examples\n")
    
    # Run examples
    example_gpt4_simple_generation()
    example_gpt4_chat()
    example_gpt4_comparison()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
