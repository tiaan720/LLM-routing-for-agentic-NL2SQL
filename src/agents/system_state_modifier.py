import logging
from typing import Any, Dict, List, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage


def create_state_modifier(
    system_prompt: str, retriever=None, retriever_id: Optional[str] = None
):
    def convert_to_messages(state_messages) -> List[BaseMessage]:
        """Convert tuple messages to BaseMessage objects."""
        if not state_messages:
            return []

        messages = []
        for msg in state_messages:
            if isinstance(msg, BaseMessage):
                messages.append(msg)
            elif isinstance(msg, tuple):
                role, content = msg
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
        return messages

    def get_last_human_message(state: Dict[str, Any]) -> Optional[BaseMessage]:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if last_message.type == "human":
            return last_message
        return None

    def modify_state(state: Dict[str, Any]) -> List[BaseMessage]:
        messages = convert_to_messages(state["messages"])
        enhanced_system_prompt = system_prompt

        last_human_message = get_last_human_message(state)
        if last_human_message and retriever is not None:
            # Retrieve examples using the retriever
            examples_retrieved = retriever.invoke(last_human_message.content)

            # Format the retrieved examples
            if examples_retrieved:
                retrieved_examples_text = "\n".join(
                    [
                        f"Query: {doc.page_content}\n" f"SQL: {doc.metadata['query']}\n"
                        for doc in examples_retrieved
                    ]
                )
                enhanced_system_prompt += f"\n\nHere are some example queries for reference:\n{retrieved_examples_text}"

        if "system_messages" in locals():
            system_messages.append(SystemMessage(content=enhanced_system_prompt))
        else:
            system_messages = [SystemMessage(content=enhanced_system_prompt)]

        # Add all system messages at the beginning
        messages = system_messages + messages

        return messages

    return modify_state
