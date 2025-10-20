"""
Global LLM Configuration and Instance
Provides a global LLM instance that can be accessed from anywhere in the application
"""

import logging
import os
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
from core.config import settings

logger = logging.getLogger(__name__)


class GlobalLLMService:
    """Global LLM service with singleton pattern for application-wide access."""
    
    _instance: Optional['GlobalLLMService'] = None
    _client: Optional[AsyncOpenAI] = None
    
    def __new__(cls) -> 'GlobalLLMService':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the LLM service if not already initialized."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_client()
    
    def _setup_client(self) -> None:
        """Set up the OpenAI client based on current settings."""
        try:
            if settings.LLM_PROVIDER == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                self._client = AsyncOpenAI(
                    api_key=api_key,
                    timeout=60.0,
                    max_retries=3
                )
            elif settings.LLM_PROVIDER == "azure_openai":
                azure_key = os.getenv("AZURE_OPENAI_API_KEY")
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                self._client = AsyncOpenAI(
                    api_key=azure_key,
                    azure_endpoint=azure_endpoint,
                    api_version="2024-02-01",
                    timeout=60.0,
                    max_retries=3
                )
            else:
                logger.warning(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
                self._client = None
            
            if self._client:
                logger.info(f"Global LLM service initialized with provider: {settings.LLM_PROVIDER}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self._client = None
    
    @property
    def client(self) -> Optional[AsyncOpenAI]:
        """Get the OpenAI client instance."""
        return self._client
    
    @property
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self._client is not None
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        *,
        is_legal_query: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a chat completion using the global LLM client.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            is_legal_query: Whether this is a legal query (uses lower temperature)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Override default model
            
        Returns:
            Generated response text
            
        Raises:
            RuntimeError: If LLM client is not available
        """
        if not self._client:
            raise RuntimeError("LLM client is not available. Check configuration.")
        
        # Determine parameters
        temp = temperature or (
            settings.LLM_TEMPERATURE_LEGAL if is_legal_query 
            else settings.LLM_TEMPERATURE_DEFAULT
        )
        tokens = max_tokens or settings.LLM_MAX_OUTPUT_TOKENS
        model_name = model or settings.LLM_MODEL
        
        try:
            token_arg = {("max_completion_tokens" if model_name.startswith("gpt-5") else "max_tokens"): tokens}
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temp,
                **token_arg,
            }
            if not model_name.startswith("gpt-5"):
                params.update({
                    "top_p": settings.LLM_TOP_P,
                    "presence_penalty": settings.LLM_PRESENCE_PENALTY,
                    "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
                })
            response = await self._client.chat.completions.create(**params)
            
            content = response.choices[0].message.content
            if content is None:
                logger.warning("LLM returned empty response")
                return ""
            
            return content
            
        except Exception as e:
            logger.error(f"LLM chat completion failed with model {model_name}: {str(e)}")
            
            # Try fallback model if available
            if settings.LLM_MODEL_FALLBACK and model_name == settings.LLM_MODEL:
                logger.info(f"Attempting fallback to {settings.LLM_MODEL_FALLBACK}")
                try:
                    fb_token_arg = {("max_completion_tokens" if settings.LLM_MODEL_FALLBACK.startswith("gpt-5") else "max_tokens"): tokens}
                    fb_params = {
                        "model": settings.LLM_MODEL_FALLBACK,
                        "messages": messages,
                        "temperature": temp,
                        **fb_token_arg,
                    }
                    if not settings.LLM_MODEL_FALLBACK.startswith("gpt-5"):
                        fb_params.update({
                            "top_p": settings.LLM_TOP_P,
                            "presence_penalty": settings.LLM_PRESENCE_PENALTY,
                            "frequency_penalty": settings.LLM_FREQUENCY_PENALTY,
                        })
                    response = await self._client.chat.completions.create(**fb_params)
                    
                    content = response.choices[0].message.content
                    if content is None:
                        logger.warning("Fallback LLM returned empty response")
                        return ""
                    
                    logger.info(f"Successfully used fallback model {settings.LLM_MODEL_FALLBACK}")
                    return content
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback model {settings.LLM_MODEL_FALLBACK} also failed: {str(fallback_error)}")
            
            raise
    
    async def generate_embeddings(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: Text to embed
            model: Override default embedding model
            
        Returns:
            List of embedding values
            
        Raises:
            RuntimeError: If LLM client is not available
        """
        if not self._client:
            raise RuntimeError("LLM client is not available. Check configuration.")
        
        model_name = model or settings.EMBEDDING_MODEL
        
        try:
            response = await self._client.embeddings.create(
                model=model_name,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information."""
        return {
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "temperature_default": settings.LLM_TEMPERATURE_DEFAULT,
            "temperature_legal": settings.LLM_TEMPERATURE_LEGAL,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "is_available": self.is_available,
            "has_api_key": bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
        }


# Global instance - this will be created when the module is imported
llm_service = GlobalLLMService()


def get_llm_service() -> GlobalLLMService:
    """Get the global LLM service instance."""
    return llm_service


def get_llm_client() -> Optional[AsyncOpenAI]:
    """Get the global LLM client directly."""
    return llm_service.client


# Convenience functions for backward compatibility and ease of use
async def chat_completion(
    messages: List[Dict[str, str]], 
    *,
    is_legal_query: bool = False,
    **kwargs
) -> str:
    """Convenience function for chat completion using global LLM service."""
    return await llm_service.chat_completion(
        messages, 
        is_legal_query=is_legal_query, 
        **kwargs
    )


async def generate_embeddings(text: str, **kwargs) -> List[float]:
    """Convenience function for embedding generation using global LLM service."""
    return await llm_service.generate_embeddings(text, **kwargs)


def is_llm_available() -> bool:
    """Check if LLM service is available."""
    return llm_service.is_available


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration information."""
    return llm_service.get_config_info()
