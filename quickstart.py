#!/usr/bin/env python3
"""
Quick start demo script for the Language Model library
Demonstrates basic usage of both GPT-4 and sLM
"""

import sys
from language_model import LanguageModelFactory

def demo_gpt4():
    """Demo GPT-4 model"""
    print("=" * 70)
    print("GPT-4.1 DEMO")
    print("=" * 70)
    
    try:
        print("\n📝 Creating GPT-4 model instance...")
        model = LanguageModelFactory.create_gpt4()
        
        print("✓ Model created successfully!")
        print("\n💬 Example: Simple text generation")
        
        prompt = "What are the main differences between large and small language models?"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...\n")
        
        response = model.generate(prompt, max_tokens=150, temperature=0.7)
        print(f"Response:\n{response}\n")
        
        return True
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use GPT-4, please:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run this script again\n")
        return False
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install required packages:")
        print("pip install openai python-dotenv\n")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        return False


def demo_slm():
    """Demo small language model"""
    print("=" * 70)
    print("SMALL LANGUAGE MODEL (sLM) DEMO")
    print("=" * 70)
    
    try:
        print("\n📝 Creating sLM model instance (this may take a moment)...")
        model = LanguageModelFactory.create_slm(model_name="distilgpt2")
        
        print("✓ Model loaded successfully!")
        print("\n💬 Example: Simple text generation")
        
        prompt = "The future of artificial intelligence is"
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...\n")
        
        response = model.generate(prompt, max_tokens=50, temperature=0.7)
        print(f"Response:\n{response}\n")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install required packages:")
        print("pip install transformers torch python-dotenv\n")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        return False


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("LANGUAGE MODEL LIBRARY - QUICK START DEMO")
    print("=" * 70)
    print("\nThis demo showcases both GPT-4.1 and small Language Models (sLM)")
    print("=" * 70 + "\n")
    
    # Demo sLM first (doesn't require API key)
    print("Starting with sLM (runs locally, no API key needed)...\n")
    slm_success = demo_slm()
    
    # Demo GPT-4
    print("\nNow trying GPT-4 (requires API key)...\n")
    gpt4_success = demo_gpt4()
    
    # Summary
    print("=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"sLM Demo:  {'✓ Success' if slm_success else '✗ Failed (see above)'}")
    print(f"GPT-4 Demo: {'✓ Success' if gpt4_success else '✗ Failed (see above)'}")
    print("=" * 70)
    
    if slm_success or gpt4_success:
        print("\n✨ For more examples, check out:")
        print("   - example_slm.py (sLM examples)")
        print("   - example_gpt4.py (GPT-4 examples)")
        print("   - example_comparison.py (compare both models)")
        print("\n📚 See README.md for full documentation\n")
        return 0
    else:
        print("\n⚠️  Please install dependencies and configure API keys")
        print("   See README.md for setup instructions\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
