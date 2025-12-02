import logging
from typing import Callable, Dict, List, Optional, Union

from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Centralized registry for model providers with auto-discovery."""

    _providers: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model provider.

        Args:
            name: Provider name (e.g., 'openai', 'vertex_ai')
        """

        def decorator(func: Callable):
            cls._providers[name] = func
            logger.debug(f"Registered model provider: {name}")
            return func

        return decorator

    @classmethod
    def create_model(cls, provider: str, **kwargs) -> Runnable:
        """Create a model instance using the registry.

        Args:
            provider: Provider name
            **kwargs: Model configuration parameters

        Returns:
            Runnable: Model instance

        Raises:
            ValueError: If provider is not registered
        """
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

        return cls._providers[provider](**kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """Get list of registered providers."""
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, provider: str) -> bool:
        """Check if a provider is registered."""
        return provider in cls._providers


def _ensure_model_name(model: Runnable, model_kwargs: Dict) -> Runnable:
    """Ensure model has a model_name attribute for cost tracking.

    Args:
        model: Model instance
        model_kwargs: Original model configuration

    Returns:
        Runnable: Model with model_name attribute
    """
    # Skip if already has model_name
    if hasattr(model, "model_name") and model.model_name:
        return model

    # Try to extract from model_kwargs
    if hasattr(model, "model_kwargs"):
        if "model_name" in model.model_kwargs:
            model.model_name = model.model_kwargs["model_name"]
        elif "model" in model.model_kwargs:
            model.model_name = model.model_kwargs["model"]

    # For models like ChatOllama that have a model attribute
    elif hasattr(model, "model") and model.model:
        object.__setattr__(model, "model_name", model.model)

    # Fallback to model_kwargs
    elif "model_name" in model_kwargs:
        object.__setattr__(model, "model_name", model_kwargs["model_name"])
    elif "model" in model_kwargs:
        object.__setattr__(model, "model_name", model_kwargs["model"])

    return model


def create_model(model: str, model_kwargs: Dict) -> Runnable:
    """Create a model instance.

    Args:
        model: Model provider name (e.g., 'openai', 'vertex_ai')
        model_kwargs: Model configuration parameters

    Returns:
        Runnable: Configured model instance

    Raises:
        ValueError: If provider is unknown
    """
    runnable = ModelRegistry.create_model(model, **model_kwargs)

    runnable = _ensure_model_name(runnable, model_kwargs)

    return runnable


def get_available_providers() -> List[str]:
    """Get list of available model providers."""
    return ModelRegistry.list_providers()


from src.models import (
    huggingface_model,
    ollama_models,
    openai_model,
    vertex_ai_anthropic_model,
    vertex_ai_deployment_model,
    vertex_ai_gemini_model,
    vertex_ai_meta,
    vertex_ai_mistral,
)
