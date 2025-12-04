"""
Comparison example between GPT-4 and small Language Models (sLM)
"""

from language_model import LanguageModelFactory

def compare_models():
    """Compare responses from GPT-4 and sLM on the same prompt"""
    print("=" * 60)
    print("GPT-4 vs sLM Comparison")
    print("=" * 60)
    
    prompt = "The benefits of artificial intelligence include"
    print(f"\nPrompt: {prompt}\n")
    
    # Test sLM
    print("-" * 60)
    print("Small Language Model (distilgpt2)")
    print("-" * 60)
    try:
        slm = LanguageModelFactory.create_slm(model_name="distilgpt2")
        slm_response = slm.generate(prompt, max_tokens=60, temperature=0.7)
        print(f"\nResponse:\n{slm_response}\n")
    except Exception as e:
        print(f"Error with sLM: {e}\n")
    
    # Test GPT-4
    print("-" * 60)
    print("GPT-4.1 Model")
    print("-" * 60)
    try:
        gpt4 = LanguageModelFactory.create_gpt4()
        gpt4_response = gpt4.generate(prompt, max_tokens=100, temperature=0.7)
        print(f"\nResponse:\n{gpt4_response}\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY in your .env file\n")
    except Exception as e:
        print(f"Error with GPT-4: {e}\n")
    
    print("=" * 60)
    print("Comparison Summary:")
    print("=" * 60)
    print("""
Key Differences:

1. Model Size:
   - sLM (distilgpt2): ~82M parameters, runs locally
   - GPT-4: Much larger, requires API access

2. Response Quality:
   - sLM: Faster, lower resource requirements, good for simple tasks
   - GPT-4: More coherent, contextually aware, better for complex tasks

3. Cost:
   - sLM: Free to run locally (requires compute resources)
   - GPT-4: Requires API key and usage fees

4. Use Cases:
   - sLM: Edge devices, quick prototyping, resource-constrained environments
   - GPT-4: Production applications requiring high quality, complex reasoning
    """)


if __name__ == "__main__":
    compare_models()
