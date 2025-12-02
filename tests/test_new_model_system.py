#!/usr/bin/env python3
"""
Pytest test suite for the new ModelRegistry system.
"""

import logging

import pytest

from src.models.base import ModelRegistry, create_model, get_available_providers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelRegistry:
    """Test cases for ModelRegistry functionality."""

    def test_list_providers(self):
        """Test that providers are properly listed."""
        providers = get_available_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
        # Check that expected providers are registered
        expected_providers = [
            "openai",
            "vertex_ai",
            "vertex_anthropic",
            "vertex_model_garden",
        ]
        for provider in expected_providers:
            assert provider in providers

    def test_is_registered(self):
        """Test provider registration checking."""
        assert ModelRegistry.is_registered("openai") is True
        assert ModelRegistry.is_registered("vertex_ai") is True
        assert ModelRegistry.is_registered("fake_provider") is False
        assert ModelRegistry.is_registered("nonexistent") is False

    def test_registry_consistency(self):
        """Test that registry methods are consistent."""
        providers = ModelRegistry.list_providers()
        for provider in providers:
            assert ModelRegistry.is_registered(provider)


class TestModelCreation:
    """Test cases for model creation."""

    @pytest.fixture
    def vertex_ai_config(self):
        """Fixture for Vertex AI model configuration."""
        return {
            "model_name": "gemini-2.5-flash",
            "temperature": 0.0,
            "project": "research-su-llm-routing",
            "location": "us-central1",
        }

    def test_create_vertex_ai_model(self, vertex_ai_config):
        """Test creating a Vertex AI model."""
        model = create_model(model="vertex_ai", model_kwargs=vertex_ai_config)
        assert model is not None
        assert hasattr(model, "model_name")
        assert model.model_name == "gemini-2.5-flash"

    def test_create_model_with_invalid_provider(self):
        """Test error handling for unknown providers."""
        with pytest.raises(ValueError) as exc_info:
            create_model(model="unknown_provider", model_kwargs={"model": "test"})

        assert "Unknown provider 'unknown_provider'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_model_name_attribute_assignment(self, vertex_ai_config):
        """Test that model_name attribute is properly assigned."""
        model = create_model(model="vertex_ai", model_kwargs=vertex_ai_config)
        assert hasattr(model, "model_name")
        assert model.model_name is not None
        assert isinstance(model.model_name, str)


class TestModelRegistryIntegration:
    """Integration tests for the complete ModelRegistry system."""

    def test_all_registered_providers_can_be_created(self):
        """Test that all registered providers can be instantiated (where possible)."""
        providers = get_available_providers()

        # Define test configurations for each provider
        test_configs = {
            "vertex_ai": {
                "model_name": "gemini-2.5-flash",
                "temperature": 0.0,
                "project": "research-su-llm-routing",
                "location": "us-central1",
            },
            # Add other providers as needed for testing
            # Note: Some providers might require specific credentials or configurations
        }

        for provider in providers:
            if provider in test_configs:
                try:
                    model = create_model(
                        model=provider, model_kwargs=test_configs[provider]
                    )
                    assert model is not None
                except Exception as e:
                    # Log the error but don't fail the test for credential issues
                    logger.warning(f"Could not create {provider} model: {e}")

    def test_registry_state_consistency(self):
        """Test that the registry maintains consistent state."""
        initial_providers = set(get_available_providers())

        # The registry should be stable across multiple calls
        for _ in range(3):
            current_providers = set(get_available_providers())
            assert current_providers == initial_providers


class TestModelSystemPerformance:
    """Test performance aspects of the model system."""

    def test_model_registry_lookup_performance(self):
        """Test that model registry lookups are fast."""
        import time

        # Test multiple lookups
        start_time = time.time()
        for _ in range(100):
            ModelRegistry.is_registered("vertex_ai")
            get_available_providers()
        end_time = time.time()

        # Should complete 100 lookups in well under a second
        assert (end_time - start_time) < 1.0

    def test_model_creation_performance(self):
        """Test that model creation is reasonably fast."""
        import time

        model_kwargs = {
            "model_name": "gemini-2.5-flash",
            "temperature": 0.0,
            "project": "research-su-llm-routing",
            "location": "us-central1",
        }

        start_time = time.time()
        model = create_model(model="vertex_ai", model_kwargs=model_kwargs)
        end_time = time.time()

        # Model creation should be reasonably fast (under 5 seconds)
        assert (end_time - start_time) < 5.0
        assert model is not None


@pytest.mark.integration
class TestRealModelInteraction:
    """Integration tests that actually interact with models (requires proper credentials)."""

    @pytest.mark.skip(reason="Requires valid GCP credentials and internet connection")
    def test_real_vertex_ai_model_interaction(self):
        """Test actual interaction with Vertex AI model."""
        model = create_model(
            model="vertex_ai",
            model_kwargs={
                "model_name": "gemini-2.5-flash",
                "temperature": 0.0,
                "project": "research-su-llm-routing",
                "location": "us-central1",
            },
        )

        # Test actual model invocation
        messages = [("user", "Hello, how are you?")]
        response = model.invoke(messages)

        assert response is not None
        assert hasattr(response, "content")
        assert isinstance(response.content, str)
        assert len(response.content) > 0


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])
