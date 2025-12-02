"""
Store-based cost tracking system for LangGraph agents.

This module provides a robust cost tracking system that uses LangGraph's InMemoryStore
to persist cost information across the entire agent execution graph, including nested
agents and tool calls.
"""

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from uuid import uuid4

from langgraph.store.base import BaseStore

from src.cost_calc.cost_tracker import (
    calculate_time_based_costs,
    calculate_token_costs,
    extract_model_from_tool_calls,
)
from src.utils.logger import logger


class StoreCostTracker:
    """
    A cost tracking system that uses LangGraph's store to persist costs
    across the entire agent execution graph.
    """

    COST_NAMESPACE = "costs"
    SESSION_KEY_PREFIX = "session_"
    AGENT_KEY_PREFIX = "agent_"

    def __init__(self, store: BaseStore, session_id: Optional[str] = None):
        """
        Initialize the store-based cost tracker.

        Args:
            store: LangGraph store instance (shared across agents)
            session_id: Unique session identifier for this execution
        """
        self.store = store
        self.session_id = session_id or str(uuid4())
        self.session_key = f"{self.SESSION_KEY_PREFIX}{self.session_id}"

        # Initialize session in store if not exists
        self._init_session()

    def _init_session(self):
        """Initialize the cost tracking session in the store."""
        session_data = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "total_input_cost": 0.0,
            "total_output_cost": 0.0,
            "agent_costs": {},
            "model_usage": {},
            "execution_path": [],
        }

        # Store session data
        try:
            self.store.put(
                namespace=(self.COST_NAMESPACE,),
                key=self.session_key,
                value=session_data,
            )
        except Exception as e:
            logger.error(f"Failed to initialize session in store: {e}")
            raise

    def add_agent_cost(
        self,
        agent_name: str,
        model_name: str,
        input_cost: float,
        output_cost: float,
        metadata: Optional[Dict] = None,
    ):
        """
        Add cost information for a specific agent execution.

        Args:
            agent_name: Name/identifier of the agent
            model_name: Model used by the agent
            input_cost: Input token cost
            output_cost: Output token cost
            metadata: Additional metadata about the execution
        """
        # Get current session data
        session_data = self.store.get((self.COST_NAMESPACE,), self.session_key)
        if not session_data:
            self._init_session()
            session_data = self.store.get((self.COST_NAMESPACE,), self.session_key)

        # Update totals
        session_data.value["total_input_cost"] += input_cost
        session_data.value["total_output_cost"] += output_cost

        # Track agent-specific costs
        agent_key = f"{agent_name}_{model_name}"
        if agent_key not in session_data.value["agent_costs"]:
            session_data.value["agent_costs"][agent_key] = {
                "agent_name": agent_name,
                "model_name": model_name,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "call_count": 0,
                "metadata": [],
            }

        agent_costs = session_data.value["agent_costs"][agent_key]
        agent_costs["input_cost"] += input_cost
        agent_costs["output_cost"] += output_cost
        agent_costs["call_count"] += 1

        if metadata:
            agent_costs["metadata"].append({"timestamp": time.time(), **metadata})

        # Track model usage
        if model_name not in session_data.value["model_usage"]:
            session_data.value["model_usage"][model_name] = {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "call_count": 0,
            }

        model_usage = session_data.value["model_usage"][model_name]
        model_usage["input_cost"] += input_cost
        model_usage["output_cost"] += output_cost
        model_usage["call_count"] += 1

        # Add to execution path
        session_data.value["execution_path"].append(
            {
                "timestamp": time.time(),
                "agent_name": agent_name,
                "model_name": model_name,
                "input_cost": input_cost,
                "output_cost": output_cost,
            }
        )

        # Update store
        self.store.put(
            namespace=(self.COST_NAMESPACE,),
            key=self.session_key,
            value=session_data.value,
        )

        logger.debug(
            f"Added costs for {agent_name} using {model_name}: "
            f"input=${input_cost:.8f}, output=${output_cost:.8f}"
        )

    def get_session_costs(self) -> Dict:
        """Get all cost information for the current session."""
        session_data = self.store.get((self.COST_NAMESPACE,), self.session_key)
        if not session_data:
            return {}
        return session_data.value

    def get_total_costs(self) -> Tuple[float, float, float]:
        """
        Get total costs for the session.

        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        session_data = self.get_session_costs()
        if not session_data:
            return 0.0, 0.0, 0.0

        input_cost = session_data.get("total_input_cost", 0.0)
        output_cost = session_data.get("total_output_cost", 0.0)
        return input_cost, output_cost, input_cost + output_cost

    def get_model_breakdown(self) -> Dict[str, Dict]:
        """Get cost breakdown by model."""
        session_data = self.get_session_costs()
        return session_data.get("model_usage", {})

    def get_agent_breakdown(self) -> Dict[str, Dict]:
        """Get cost breakdown by agent."""
        session_data = self.get_session_costs()
        return session_data.get("agent_costs", {})

    def finalize_session(self):
        """Finalize the session and log summary."""
        session_data = self.get_session_costs()
        if not session_data:
            logger.warning("No session data found to finalize")
            return

        # Update end time
        session_data["end_time"] = time.time()
        session_data["duration"] = session_data["end_time"] - session_data["start_time"]

        self.store.put(
            namespace=(self.COST_NAMESPACE,), key=self.session_key, value=session_data
        )

        # Log comprehensive summary
        self._log_cost_summary(session_data)

    def _log_cost_summary(self, session_data: Dict):
        """Log a concise cost summary."""
        total_input = session_data.get("total_input_cost", 0)
        total_output = session_data.get("total_output_cost", 0)
        total_cost = total_input + total_output
        duration = session_data.get("duration", 0)

        logger.info(
            f"Session {session_data['session_id'][:8]}: "
            f"${total_cost:.6f} total ({duration:.1f}s)"
        )

        # Log model breakdown concisely
        models = session_data.get("model_usage", {})
        if len(models) > 1:
            model_costs = [
                f"{model}: ${usage['input_cost'] + usage['output_cost']:.6f}"
                for model, usage in models.items()
            ]
            logger.info(f"  Models: {', '.join(model_costs)}")


def process_agent_stream_with_store(
    agent_stream: Generator,
    store: BaseStore,
    agent_name: str,
    model_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[
    List[Dict], Optional[float], Optional[float], Optional[str], StoreCostTracker
]:
    """
    Process agent stream and track costs in the store.

    Args:
        agent_stream: The agent stream to process
        store: LangGraph store instance
        agent_name: Name of the agent being tracked
        model_name: Default model name
        session_id: Session ID (will be generated if not provided)

    Returns:
        Tuple of (responses, input_cost, output_cost, selected_model, cost_tracker)
    """
    cost_tracker = StoreCostTracker(store, session_id)

    responses = []
    total_input_cost = 0
    total_output_cost = 0
    selected_model = None
    has_valid_costs = False

    for step in agent_stream:
        responses.append(step)

        # Extract model name if not already found
        step_model = extract_model_from_tool_calls(step) or model_name
        if step_model:
            selected_model = step_model

        # Process any step that contains messages with usage metadata
        for step_name, step_data in step.items():
            if isinstance(step_data, dict) and "messages" in step_data:
                for message in step_data["messages"]:
                    costs = {"input_cost": None, "output_cost": None}

                    # Check if this is a vertex model garden deployment (time-based cost)
                    if hasattr(
                        message, "response_metadata"
                    ) and message.response_metadata.get(
                        "is_vertex_model_garden", False
                    ):
                        costs = calculate_time_based_costs(message.response_metadata)
                    # Otherwise use token-based cost calculation
                    elif hasattr(message, "usage_metadata"):
                        costs = calculate_token_costs(
                            message.usage_metadata,
                            model_name=selected_model or model_name,
                        )

                    # Only process if we got valid costs
                    if (
                        costs["input_cost"] is not None
                        and costs["output_cost"] is not None
                    ):
                        has_valid_costs = True
                        input_cost = costs["input_cost"]
                        output_cost = costs["output_cost"]

                        total_input_cost += input_cost
                        total_output_cost += output_cost

                        # Add to store tracker
                        cost_tracker.add_agent_cost(
                            agent_name=agent_name,
                            model_name=selected_model or model_name or "unknown",
                            input_cost=input_cost,
                            output_cost=output_cost,
                            metadata={
                                "step_name": step_name,
                                "usage_metadata": getattr(
                                    message, "usage_metadata", {}
                                ),
                                "response_metadata": getattr(
                                    message, "response_metadata", {}
                                ),
                            },
                        )

    return (
        responses,
        total_input_cost if has_valid_costs else None,
        total_output_cost if has_valid_costs else None,
        selected_model,
        cost_tracker,
    )


# Convenience function for backward compatibility
def get_session_cost_summary(store: BaseStore, session_id: str) -> Dict:
    """Get cost summary for a specific session."""
    cost_tracker = StoreCostTracker(store, session_id)
    return cost_tracker.get_session_costs()
