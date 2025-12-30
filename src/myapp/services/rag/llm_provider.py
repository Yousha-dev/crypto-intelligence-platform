"""
Dynamic LLM Provider Management
Supports OpenAI, Anthropic Claude, and local models (Ollama/vLLM)
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

from django.conf import settings
 
logger = logging.getLogger(__name__)
  
 
class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"
    GROQ = "groq"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.3
    timeout: int = 30
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Tuple of (response_text, tokens_used)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'provider': self.config.provider.value,
            'model': self.config.model_name,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature
        }


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialize()
    
    def _initialize(self):
        try:
            from openai import OpenAI
            
            api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found")
                self._client = None
                return
            
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            logger.info(f"OpenAI client initialized with model: {self.config.model_name}")
            
        except ImportError:
            logger.error("OpenAI package not installed")
            self._client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self._client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        if not self._client:
            raise RuntimeError("OpenAI client not initialized")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params
            )
            
            return response.choices[0].message.content, response.usage.total_tokens
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        return self._client is not None


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialize()
    
    def _initialize(self):
        try:
            import anthropic
            
            api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("Anthropic API key not found")
                self._client = None
                return
            
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Anthropic client initialized with model: {self.config.model_name}")
            
        except ImportError:
            logger.error("Anthropic package not installed. Run: pip install anthropic")
            self._client = None
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            self._client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        if not self._client:
            raise RuntimeError("Anthropic client not initialized")
        
        try:
            message = self._client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "You are a helpful cryptocurrency news analyst.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text
            tokens_used = message.usage.input_tokens + message.usage.output_tokens
            
            return response_text, tokens_used
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        return self._client is not None


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self._initialize()
    
    def _initialize(self):
        try:
            import requests
            
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                if self.config.model_name in available_models or any(self.config.model_name in m for m in available_models):
                    self._client = True
                    logger.info(f"Ollama client initialized with model: {self.config.model_name}")
                else:
                    logger.warning(f"Model {self.config.model_name} not found in Ollama. Available: {available_models}")
                    self._client = None
            else:
                self._client = None
                
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        if not self._client:
            raise RuntimeError("Ollama client not initialized")
        
        import requests
        
        try:
            # ✅ Use chat endpoint instead of generate
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.base_url}/api/chat",  # Changed from /api/generate
                json={
                    "model": self.config.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                message_content = result.get('message', {}).get('content', '')
                tokens = result.get('eval_count', 0)
                return message_content, tokens
            else:
                raise RuntimeError(f"Ollama error: {response.text}")
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        return self._client is not None


class GroqClient(BaseLLMClient):
    """Groq API client for fast inference"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialize()
    
    def _initialize(self):
        try:
            from groq import Groq
            
            api_key = self.config.api_key or os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.warning("Groq API key not found")
                self._client = None
                return
            
            self._client = Groq(api_key=api_key)
            logger.info(f"Groq client initialized with model: {self.config.model_name}")
            
        except ImportError:
            logger.error("Groq package not installed. Run: pip install groq")
            self._client = None
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")
            self._client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        if not self._client:
            raise RuntimeError("Groq client not initialized")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content, response.usage.total_tokens
            
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        return self._client is not None


class LLMProviderManager:
    """
    Dynamic LLM provider manager with failover support
    """
    
    # Default model configurations
    DEFAULT_CONFIGS = {
        LLMProvider.OPENAI: LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.3
        ),
        LLMProvider.ANTHROPIC: LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.3
        ),
        LLMProvider.OLLAMA: LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.1",
            max_tokens=1000,
            temperature=0.3
        ),
        LLMProvider.GROQ: LLMConfig(
            provider=LLMProvider.GROQ,
            model_name="llama-3.1-70b-versatile",
            max_tokens=1000,
            temperature=0.3
        )
    }
    
    def __init__(self, 
                 primary_provider: LLMProvider = LLMProvider.OPENAI,
                 fallback_order: Optional[List[LLMProvider]] = None,
                 custom_configs: Optional[Dict[LLMProvider, LLMConfig]] = None):
        """
        Initialize LLM provider manager
        
        Args:
            primary_provider: Primary LLM provider to use
            fallback_order: Order of providers to try on failure
            custom_configs: Custom configurations for providers
        """
        self.primary_provider = primary_provider
        self.fallback_order = fallback_order or [
            LLMProvider.OPENAI,
            LLMProvider.GROQ,
            LLMProvider.ANTHROPIC,
            LLMProvider.OLLAMA
        ]
        
        # Merge custom configs with defaults
        self.configs = {**self.DEFAULT_CONFIGS}
        if custom_configs:
            self.configs.update(custom_configs)
        
        # Initialize clients
        self.clients: Dict[LLMProvider, BaseLLMClient] = {}
        self._initialize_clients()
        
        # Track current active provider
        self.active_provider: Optional[LLMProvider] = None
        self._select_active_provider()
    
    def _initialize_clients(self):
        """Initialize all configured LLM clients"""
        client_classes = {
            LLMProvider.OPENAI: OpenAIClient,
            LLMProvider.ANTHROPIC: AnthropicClient,
            LLMProvider.OLLAMA: OllamaClient,
            LLMProvider.GROQ: GroqClient
        }
        
        for provider, config in self.configs.items():
            if provider in client_classes:
                try:
                    client = client_classes[provider](config)
                    self.clients[provider] = client
                    logger.info(f"Initialized {provider.value} client: available={client.is_available()}")
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider.value}: {e}")
    
    def _select_active_provider(self):
        """Select the best available provider"""
        # Try primary first
        if self.primary_provider in self.clients:
            if self.clients[self.primary_provider].is_available():
                self.active_provider = self.primary_provider
                logger.info(f"Using primary provider: {self.primary_provider.value}")
                return
        
        # Try fallback order
        for provider in self.fallback_order:
            if provider in self.clients and self.clients[provider].is_available():
                self.active_provider = provider
                logger.info(f"Using fallback provider: {provider.value}")
                return
        
        logger.error("No LLM provider available!")
        self.active_provider = None
    
    def generate(self, prompt: str, 
                 system_prompt: Optional[str] = None,
                 provider: Optional[LLMProvider] = None) -> Tuple[str, int, str]:
        """
        Generate response using available LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            provider: Specific provider to use (optional)
            
        Returns:
            Tuple of (response_text, tokens_used, provider_used)
        """
        # Use specified provider or active provider
        target_provider = provider or self.active_provider
        
        if not target_provider:
            raise RuntimeError("No LLM provider available")
        
        # Try target provider first
        providers_to_try = [target_provider] + [
            p for p in self.fallback_order 
            if p != target_provider and p in self.clients
        ]
        
        last_error = None
        for prov in providers_to_try:
            if prov not in self.clients or not self.clients[prov].is_available():
                continue
            
            try:
                response, tokens = self.clients[prov].generate(prompt, system_prompt)
                return response, tokens, prov.value
                
            except Exception as e:
                logger.warning(f"Provider {prov.value} failed: {e}")
                last_error = e
                continue
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [
            provider.value 
            for provider, client in self.clients.items() 
            if client.is_available()
        ]
    
    def switch_provider(self, provider: LLMProvider) -> bool:
        """
        Switch to a different provider
        
        Args:
            provider: Provider to switch to
            
        Returns:
            True if switch successful
        """
        if provider not in self.clients:
            logger.error(f"Provider {provider.value} not configured")
            return False
        
        if not self.clients[provider].is_available():
            logger.error(f"Provider {provider.value} not available")
            return False
        
        self.active_provider = provider
        logger.info(f"Switched to provider: {provider.value}")
        return True
    
    def update_config(self, provider: LLMProvider, config: LLMConfig):
        """Update configuration for a provider"""
        self.configs[provider] = config
        
        # Reinitialize client
        client_classes = {
            LLMProvider.OPENAI: OpenAIClient,
            LLMProvider.ANTHROPIC: AnthropicClient,
            LLMProvider.OLLAMA: OllamaClient,
            LLMProvider.GROQ: GroqClient
        }
        
        if provider in client_classes:
            self.clients[provider] = client_classes[provider](config)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            'active_provider': self.active_provider.value if self.active_provider else None,
            'providers': {
                provider.value: {
                    'available': client.is_available(),
                    'model': client.get_model_info()
                }
                for provider, client in self.clients.items()
            }
        }


# Singleton instance
_llm_manager_instance = None


def get_llm_manager() -> LLMProviderManager:
    """Get singleton LLM provider manager"""
    global _llm_manager_instance
    
    if _llm_manager_instance is None:
        # Get configuration from Django settings
        primary = getattr(settings, 'LLM_PRIMARY_PROVIDER', 'openai')
        
        try:
            primary_provider = LLMProvider(primary)
        except ValueError:
            primary_provider = LLMProvider.OPENAI
        
        _llm_manager_instance = LLMProviderManager(
            primary_provider=primary_provider
        )
    
    return _llm_manager_instance