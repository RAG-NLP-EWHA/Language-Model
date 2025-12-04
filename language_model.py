"""
Language Model Integration Module
Supports both GPT-4.1 (via OpenAI API) and small Language Models (sLM)
"""

import os
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LanguageModel(ABC):
    """Abstract base class for language models"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text based on the given prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate chat response based on message history"""
        pass


class GPT4Model(LanguageModel):
    """GPT-4.1 model implementation using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-turbo-preview"):
        """
        Initialize GPT-4 model
        
        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            model_name: Name of the GPT-4 model to use (default: gpt-4-turbo-preview)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install it with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text based on the given prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens, temperature)
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate chat response based on message history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating response from GPT-4: {str(e)}")


class SmallLanguageModel(LanguageModel):
    """Small Language Model (sLM) implementation using transformers"""
    
    def __init__(self, model_name: str = "distilgpt2", device: Optional[str] = None):
        """
        Initialize small language model
        
        Args:
            model_name: Name of the model from HuggingFace (default: distilgpt2)
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("transformers and torch packages are required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading {model_name} model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully!")
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text based on the given prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate chat response based on message history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Format messages into a single prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        
        generated = self.generate(prompt, max_tokens, temperature)
        
        # Extract only the assistant's response
        if "Assistant:" in generated:
            response = generated.split("Assistant:")[-1].strip()
        else:
            response = generated.replace(prompt, "").strip()
        
        return response


class LanguageModelFactory:
    """Factory class for creating language model instances"""
    
    @staticmethod
    def create_gpt4(api_key: Optional[str] = None, model_name: str = "gpt-4-turbo-preview") -> GPT4Model:
        """
        Create a GPT-4 model instance
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the GPT-4 model
            
        Returns:
            GPT4Model instance
        """
        return GPT4Model(api_key=api_key, model_name=model_name)
    
    @staticmethod
    def create_slm(model_name: str = "distilgpt2", device: Optional[str] = None) -> SmallLanguageModel:
        """
        Create a small language model instance
        
        Args:
            model_name: Name of the model from HuggingFace
            device: Device to run the model on
            
        Returns:
            SmallLanguageModel instance
        """
        return SmallLanguageModel(model_name=model_name, device=device)
