import concurrent.futures
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.auth
import requests
from google.auth.transport.requests import Request
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Unified model configuration schema."""

    model_name: str
    provider: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 0
    stop: Optional[str] = None
    base_url: Optional[str] = None
    project: Optional[str] = None
    location: Optional[str] = None

    def to_provider_format(self) -> Dict[str, Any]:
        """Convert to provider-specific format."""
        config = {}

        if self.provider in ["vertex_ai"]:
            config["model_name"] = self.model_name
        else:
            config["model"] = self.model_name

        config.update(
            {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_retries": self.max_retries,
                "stop": self.stop,
            }
        )

        if self.base_url:
            config["base_url"] = self.base_url
        if self.project:
            config["project"] = self.project
        if self.location:
            config["location"] = self.location

        if self.provider == "vertex_meta":
            config["append_tools_to_system_message"] = True

        return {k: v for k, v in config.items() if v is not None}


class ModelConfigLoader:
    """Handles model discovery, testing, and configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = (
            config_path
            or Path(__file__).parent.parent.parent / "configs" / "llm_model_config.json"
        )
        if not self.config_path.exists():
            try:
                with open(self.config_path, "w") as f:
                    json.dump({"models": {}}, f, indent=2)
                logger.info(f"Created config file: {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to create config file: {e}")
        self.gcp_project = self._init_gcp()
        self.gcp_locations = [
            "us-central1",
            "us-east1",
            "us-east4",
            "us-west1",
            "europe-west1",
            "asia-southeast1",
            "us-east5",
        ]

    def _init_gcp(self) -> str:
        """Initialize GCP authentication."""
        try:
            user_creds, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if not user_creds.valid:
                user_creds.refresh(Request())
            return project
        except Exception:
            return "research-su-llm-routing"

    def get_static_configs(self) -> List[ModelConfig]:
        """Get predefined static model configurations."""
        return [
            # OpenAI models
            # ModelConfig("gpt-4o", "openai"),
            # ModelConfig("gpt-4o-mini", "openai"),
            # Ollama models
            ModelConfig(
                "deepseek-r1:1.5b", "ollama", base_url="http://localhost:11434"
            ),
        ]

    def discover_gcp_models(self) -> List[ModelConfig]:
        """Discover and configure GCP models."""
        configs = []

        try:
            vertex_models = self._fetch_vertex_models()
            vertex_models = self._fix_model_names(vertex_models)
            categorized = self._categorize_models(vertex_models)

            for provider, models in categorized.items():
                for model_name in models:
                    # Don't pre-determine location here, let _test_model find working location
                    configs.append(
                        ModelConfig(
                            model_name=model_name,
                            provider=provider,
                            project=self.gcp_project,
                            location=None,  # Will be determined during testing
                        )
                    )
        except Exception as e:
            logger.error(f"GCP discovery failed: {e}")

        return configs

    def test_and_save_models(self, max_workers: int = 18) -> Dict[str, Any]:
        """Test all models and save successful ones."""
        self._init_config_file()

        # Get all model configurations
        static_configs = self.get_static_configs()
        gcp_configs = self.discover_gcp_models()
        all_configs = static_configs + gcp_configs

        logger.info(f"Testing {len(all_configs)} models with {max_workers} workers...")
        logger.info(
            "Each model will go through 2 tests: Basic invocation -> Dummy agent test"
        )

        successful_models = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {
                executor.submit(self._test_model, config): config
                for config in all_configs
            }

            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    if future.result():
                        successful_models.append(config)
                        self._save_model(config)
                except Exception as e:
                    logger.error(f"Failed testing {config.model_name}: {e}")

        logger.info(
            f"Successfully tested {len(successful_models)}/{len(all_configs)} models"
        )
        return self._load_config()

    def _fetch_vertex_models(self) -> List[str]:
        """Fetch Vertex AI models from LiteLLM database."""
        url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            models_data = response.json()

            vertex_models = []
            for model_name, model_info in models_data.items():
                provider = model_info.get("litellm_provider", "")
                mode = model_info.get("mode", "")
                if "vertex" in provider.lower() and mode == "chat":
                    clean_name = model_name.replace("vertex_ai/", "")
                    vertex_models.append(clean_name)

            fixed_models = self._fix_model_names(vertex_models)
            return fixed_models
        except Exception:
            return []

    def _fix_model_names(self, models: List[str]) -> List[str]:
        """Fix known model name discrepancies between LiteLLM database and actual working names."""
        model_name_mappings = {
            "meta/llama3-70b-instruct-maas": "meta/llama-3.3-70b-instruct-maas",
            # Add more mappings as needed
        }

        fixed_models = []
        for model in models:
            fixed_name = model_name_mappings.get(model, model)
            fixed_models.append(fixed_name)

        return fixed_models

    def _categorize_models(self, models: List[str]) -> Dict[str, List[str]]:
        """Categorize models by provider type."""
        categorized = {
            "vertex_ai": [],
            "vertex_anthropic": [],
            "vertex_meta": [],
            "vertex_mistral": [],
        }

        for model in models:
            model_lower = model.lower()
            if "gemini" in model_lower:
                categorized["vertex_ai"].append(model)
            elif any(k in model_lower for k in ["claude", "anthropic"]):
                categorized["vertex_anthropic"].append(model)
            elif any(k in model_lower for k in ["llama", "meta"]):
                categorized["vertex_meta"].append(model)
            elif "mistral" in model_lower:
                categorized["vertex_mistral"].append(model)

        return categorized

    def _find_first_working_location(
        self, provider: str, model_name: str
    ) -> Optional[str]:
        """Find the first working location for a model."""
        for location in self.gcp_locations:
            if self._test_model_location(provider, model_name, location):
                logger.info(
                    f"Found working location for {provider}::{model_name} -> {location}"
                )
                return location
        logger.warning(f"No working location found for {provider}::{model_name}")
        return None

    def _find_working_locations(self, provider: str, model_name: str) -> List[str]:
        """Find working locations for a model."""
        working_locations = []
        for location in self.gcp_locations:
            if self._test_model_location(provider, model_name, location):
                working_locations.append(location)
        return working_locations if working_locations else [self.gcp_locations[0]]

    def _test_model_location(
        self, provider: str, model_name: str, location: str
    ) -> bool:
        """Test if model works in specific location."""
        try:
            aiplatform.init(project=self.gcp_project, location=location)

            if provider == "vertex_ai":
                from langchain_google_vertexai import ChatVertexAI

                ChatVertexAI(
                    model_name=model_name,
                    project=self.gcp_project,
                    location=location,
                    temperature=0.0,
                )
            elif provider == "vertex_anthropic":
                from langchain_google_vertexai.model_garden import ChatAnthropicVertex

                ChatAnthropicVertex(
                    model=model_name,
                    project=self.gcp_project,
                    location=location,
                    temperature=0.0,
                )
            elif provider == "vertex_meta":
                from langchain_google_vertexai.model_garden_maas.llama import (
                    VertexModelGardenLlama,
                )

                VertexModelGardenLlama(
                    model=model_name,
                    project=self.gcp_project,
                    location=location,
                    temperature=0.0,
                    append_tools_to_system_message=True,
                )
            elif provider == "vertex_mistral":
                from langchain_google_vertexai.model_garden_maas.mistral import (
                    VertexModelGardenMistral,
                )

                VertexModelGardenMistral(
                    model=model_name,
                    project=self.gcp_project,
                    location=location,
                    temperature=0.0,
                    append_tools_to_system_message=True,
                )

            return True
        except Exception:
            return False

    def _test_model(self, config: ModelConfig) -> bool:
        """Test if a model configuration works with both basic invocation and dummy agent."""
        logger.info(f"Testing {config.provider}::{config.model_name}")

        # For GCP models without pre-determined location, test all locations
        if config.provider.startswith("vertex") and config.location is None:
            return self._test_gcp_model_with_location_fallback(config)

        # For models with pre-determined location or non-GCP models
        return self._test_model_with_config(config)

    def _test_gcp_model_with_location_fallback(self, config: ModelConfig) -> bool:
        """Test GCP model across all locations until one works for both init and invocation."""
        for location in self.gcp_locations:
            test_config = ModelConfig(
                model_name=config.model_name,
                provider=config.provider,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                max_retries=config.max_retries,
                stop=config.stop,
                base_url=config.base_url,
                project=config.project,
                location=location,
            )

            logger.info(
                f"Testing {config.provider}::{config.model_name} at location {location}"
            )

            if self._test_model_with_config(test_config):
                logger.info(
                    f"SUCCESS: {config.provider}::{config.model_name} works at {location}"
                )
                # Update the original config with the working location
                config.location = location
                return True
            else:
                logger.info(
                    f"FAIL: {config.provider}::{config.model_name} failed at {location}"
                )

        logger.error(
            f"FAIL: {config.provider}::{config.model_name} failed at all locations"
        )
        return False

    def _test_model_with_config(self, config: ModelConfig) -> bool:
        """Test if a model configuration works with both basic invocation and dummy agent."""

        # First test: Basic model invocation
        def _invoke_model():
            import os
            import sys

            # TODO: fix this import here to be better
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from src.models.base import create_model

            model = create_model(config.provider, config.to_provider_format())
            test_message = [
                (
                    "system",
                    "You are a helpful assistant. Respond with exactly: 'TEST_SUCCESS'",
                ),
                ("human", "Say TEST_SUCCESS"),
            ]
            model.invoke(test_message)
            return True

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke_model)
                future.result(timeout=40)
                logger.info(f"PASS Basic Test {config.provider}::{config.model_name}")
        except concurrent.futures.TimeoutError:
            logger.error(
                f"FAIL {config.provider}::{config.model_name} - Basic Test Timeout"
            )
            return False
        except Exception as e:
            logger.error(
                f"FAIL {config.provider}::{config.model_name} - Basic Test Error: {str(e)}"
            )
            return False

        # Second test: Dummy agent test (only if basic test passed)
        def _test_dummy_agent():
            import os
            import sys

            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from src.agents.dummy_agent import create_dummy_agent
            from src.models.base import create_model

            model = create_model(config.provider, config.to_provider_format())

            agent_kwargs = {
                "tools": ["plus_calculator"],
            }

            query = "What is 1 + 1?"
            agent = create_dummy_agent(model=model, input_query=query, **agent_kwargs)

            input_data = {"messages": [("user", query)]}
            config_data = {"configurable": {"thread_id": "test_thread"}}

            result = agent.invoke(input_data, config_data)

            if not result or "messages" not in result:
                raise Exception("Agent failed to return valid result")

            return True

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_test_dummy_agent)
                future.result(timeout=60)  # Longer timeout for agent test
                logger.info(f"PASS Agent Test {config.provider}::{config.model_name}")
                return True
        except concurrent.futures.TimeoutError:
            logger.error(
                f"FAIL {config.provider}::{config.model_name} - Agent Test Timeout"
            )
            return False
        except Exception as e:
            logger.error(
                f"FAIL {config.provider}::{config.model_name} - Agent Test Error: {str(e)}"
            )
            return False

    def _init_config_file(self):
        """Initialize empty config file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump({"models": {}}, f, indent=2)
            logger.info(f"Initialized: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to initialize config: {e}")

    def _save_model(self, config: ModelConfig):
        """Save working model to config file."""
        try:
            data = self._load_config()

            if config.provider not in data["models"]:
                data["models"][config.provider] = {}

            config_key = config.model_name
            if config.location and config.provider.startswith("vertex"):
                config_key = f"{config.model_name}@{config.location}"

            data["models"][config.provider][config_key] = config.to_provider_format()

            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Added: {config.provider}::{config_key}")
        except Exception as e:
            logger.error(f"Failed to add model: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Load config from file."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"models": {}}


def main():
    """Run the model configuration loader."""
    import time

    start_time = time.time()
    loader = ModelConfigLoader()

    try:
        static_configs = loader.get_static_configs()
        logger.info(f"Found {len(static_configs)} static configs")
        for cfg in static_configs:
            logger.info(f"  {cfg.provider}: {cfg.model_name}")

        config = loader.test_and_save_models()

        elapsed_time = time.time() - start_time
        logger.info(f"Testing complete in {elapsed_time:.2f} seconds!")

        # Show summary
        total_models = sum(len(models) for models in config.get("models", {}).values())
        logger.info(f"Total working models found: {total_models}")
        for provider, models in config.get("models", {}).items():
            if models:
                logger.info(f"  {provider}: {len(models)} models")

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
