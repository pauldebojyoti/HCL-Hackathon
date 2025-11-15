"""
LLM Integration Module
Handles integration with various Language Model APIs
"""

import os
import logging
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for different LLM providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not available. Install with: pip install groq")


@dataclass
class LLMResponse:
    """
    Standard response format for LLM outputs
    """
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate response from prompt
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM is available and configured
        """
        pass


class OpenAILLM(BaseLLM):
    """
    OpenAI GPT model integration
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI LLM
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (if not set in environment)
        """
        self.model = model
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available")
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            logger.warning("OpenAI API key not found")
        
        self.client = openai.OpenAI(api_key=openai.api_key)
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse object
        """
        try:
            # Default parameters
            params = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in ['temperature', 'max_tokens']:
                    params[key] = value
            
            response = self.client.chat.completions.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                usage=response.usage.model_dump() if response.usage else None,
                metadata={'finish_reason': response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                model=self.model,
                metadata={'error': str(e)}
            )
    
    def is_available(self) -> bool:
        """
        Check if OpenAI is available and configured
        """
        return OPENAI_AVAILABLE and openai.api_key is not None


class GroqLLM(BaseLLM):
    """
    Groq API integration for fast Llama models
    """
    
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: Optional[str] = None):
        """
        Initialize Groq LLM
        
        Args:
            model: Groq model name
            api_key: Groq API key
        """
        self.model = model
        
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not available")
        
        # Set API key
        if api_key:
            self.api_key = api_key
        elif os.getenv('GROQ_API_KEY'):
            self.api_key = os.getenv('GROQ_API_KEY')
        else:
            logger.warning("Groq API key not found")
            self.api_key = None
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate response using Groq API
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        if not self.client:
            return LLMResponse(
                content="Groq API key not configured",
                model=self.model,
                metadata={'error': 'No API key'}
            )
        
        try:
            # Create chat completion
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=kwargs.get('temperature', 0.3),  # Lower temperature for more focused responses
                max_tokens=kwargs.get('max_tokens', 500),
                top_p=kwargs.get('top_p', 0.9),
                stream=False
            )
            
            # Extract content safely
            content = response.choices[0].message.content or "No response generated"
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=response.usage.model_dump() if response.usage else None,
                metadata={'finish_reason': response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                model=self.model,
                metadata={'error': str(e)}
            )
    
    def is_available(self) -> bool:
        """
        Check if Groq is available and configured
        """
        return GROQ_AVAILABLE and self.client is not None


class HuggingFaceLLM(BaseLLM):
    """
    Hugging Face transformers model integration
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        """
        Initialize Hugging Face LLM
        
        Args:
            model_name: Hugging Face model name
            device: Device to run model on ('cpu', 'cuda', or 'auto')
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package not available")
        
        self._load_model()
    
    def _load_model(self):
        """
        Load the Hugging Face model
        """
        try:
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # Determine the right pipeline task based on model
            if "flan-t5" in self.model_name.lower():
                task = "text2text-generation"
            else:
                task = "text-generation"
            
            self.pipeline = pipeline(
                task,
                model=self.model_name,
                device_map=self.device if self.device != "auto" else None
            )
            
            logger.info("Hugging Face model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate response using Hugging Face model
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        try:
            # Check if this is a text2text generation model (like Flan-T5)
            is_text2text = (self.pipeline and 
                          hasattr(self.pipeline, 'task') and 
                          self.pipeline.task == "text2text-generation")
            
            if is_text2text:
                # Parameters for text2text-generation models
                params = {
                    'max_length': kwargs.get('max_tokens', 200),
                    'temperature': kwargs.get('temperature', 0.7),
                    'do_sample': True,
                    'top_p': 0.9,
                    'num_return_sequences': 1
                }
            else:
                # Parameters for text-generation models
                params = {
                    'max_length': min(len(prompt) + kwargs.get('max_tokens', 150), 512),
                    'temperature': kwargs.get('temperature', 0.8),
                    'do_sample': True,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.2,
                    'num_return_sequences': 1
                }
                
                # Add pad_token_id if available
                try:
                    if (self.pipeline and 
                        hasattr(self.pipeline, 'tokenizer') and 
                        self.pipeline.tokenizer and
                        hasattr(self.pipeline.tokenizer, 'eos_token_id')):
                        params['pad_token_id'] = self.pipeline.tokenizer.eos_token_id
                except (AttributeError, TypeError):
                    pass
            
            # Generate response
            if not self.pipeline:
                return LLMResponse(
                    content="Pipeline not available",
                    model=self.model_name,
                    metadata={'error': 'Pipeline not initialized'}
                )
            
            response = self.pipeline(prompt, **params)
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the input prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return LLMResponse(
                content=generated_text,
                model=self.model_name,
                metadata={'full_response': response}
            )
            
        except Exception as e:
            logger.error(f"Hugging Face generation error: {str(e)}")
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                model=self.model_name,
                metadata={'error': str(e)}
            )
    
    def is_available(self) -> bool:
        """
        Check if Hugging Face model is available
        """
        return TRANSFORMERS_AVAILABLE and self.pipeline is not None


class LLMManager:
    """
    Manager class for handling multiple LLM providers
    """
    
    def __init__(self):
        """
        Initialize LLM manager
        """
        self.llms: Dict[str, BaseLLM] = {}
        self.default_llm: Optional[str] = None
        
        # Try to initialize available LLMs
        self._initialize_llms()
    
    def _initialize_llms(self):
        """
        Initialize available LLM providers
        """
        # Try Groq first (fastest and best quality)
        if GROQ_AVAILABLE:
            try:
                groq_llm = GroqLLM("llama-3.1-8b-instant")
                if groq_llm.is_available():
                    self.llms['groq'] = groq_llm
                    self.default_llm = 'groq'
                    logger.info("Groq LLM initialized with Llama-3.1-8B")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {str(e)}")
        
        # Try OpenAI (fallback)
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY') and not self.default_llm:
            try:
                openai_llm = OpenAILLM()
                if openai_llm.is_available():
                    self.llms['openai'] = openai_llm
                    self.default_llm = 'openai'
                    logger.info("OpenAI LLM initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {str(e)}")
        
        # Try Hugging Face (last resort)
        if TRANSFORMERS_AVAILABLE and not self.default_llm:
            try:
                # Use Flan-T5 - specifically designed for instruction following and QA
                hf_llm = HuggingFaceLLM("google/flan-t5-base")
                if hf_llm.is_available():
                    self.llms['huggingface'] = hf_llm
                    if not self.default_llm:
                        self.default_llm = 'huggingface'
                    logger.info("Hugging Face LLM initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Hugging Face: {str(e)}")
    
    def add_llm(self, name: str, llm: BaseLLM):
        """
        Add a custom LLM to the manager
        
        Args:
            name: Name identifier for the LLM
            llm: LLM instance
        """
        self.llms[name] = llm
        if not self.default_llm:
            self.default_llm = name
        logger.info(f"Added LLM: {name}")
    
    def generate(self, prompt: str, llm_name: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate response using specified or default LLM
        
        Args:
            prompt: Input prompt
            llm_name: Name of LLM to use (uses default if None)
            **kwargs: Additional parameters for generation
            
        Returns:
            LLMResponse object
        """
        # Use specified LLM or default
        llm_to_use = llm_name or self.default_llm
        
        if not llm_to_use or llm_to_use not in self.llms:
            return LLMResponse(
                content="No available LLM found. Please configure an LLM provider.",
                model="none",
                metadata={'error': 'No LLM available'}
            )
        
        return self.llms[llm_to_use].generate(prompt, **kwargs)
    
    def get_available_llms(self) -> List[str]:
        """
        Get list of available LLM names
        
        Returns:
            List of available LLM identifiers
        """
        return list(self.llms.keys())
    
    def is_available(self) -> bool:
        """
        Check if any LLM is available
        
        Returns:
            True if at least one LLM is available
        """
        return len(self.llms) > 0


def create_rag_prompt(context: str, question: str) -> str:
    """
    Create a RAG prompt template optimized for better responses
    
    Args:
        context: Retrieved context from documents
        question: User question
        
    Returns:
        Formatted prompt string
    """
    # Enhanced prompt with better handling for various scenarios
    prompt = f"""You are a helpful assistant specializing in academic regulations. Use ONLY the provided information to answer the question accurately and directly.

Information:
{context}

Question: {question}

Instructions:
- Answer ONLY what is asked in the question
- Stay focused on the specific topic mentioned in the question
- Be precise and cite relevant source text from the provided information
- Do not include unrelated information or topics not asked about
- If the question is about a specific topic (like plagiarism policy, admission requirements, etc.), focus exclusively on that topic
- If the provided information does not contain relevant data to answer the question, respond with: "I couldn't find any relevant information about this topic in the knowledge base. Please try rephrasing your question or ask about topics covered in the available documents."
- If the information is partial or unclear, mention what is available and indicate what information might be missing
- Always be honest about the limitations of the available information
- Do not make up or assume information that is not explicitly provided

Answer:"""
    
    return prompt


def create_greeting_prompt() -> str:
    """
    Create a simple greeting prompt for new users
    
    Returns:
        Greeting message string
    """
    greeting = """👋 Welcome to the RAG-Powered Assistant!

                    Just ask me any question!!!!"""
    
    return greeting


def main():
    """
    Test the LLM integration
    """
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    print(f"Available LLMs: {llm_manager.get_available_llms()}")
    
    if llm_manager.is_available():
        # Test prompt
        test_prompt = "What is the capital of France?"
        
        response = llm_manager.generate(test_prompt, temperature=0.7)
        
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Metadata: {response.metadata}")
    else:
        print("No LLM available for testing")


if __name__ == "__main__":
    main()