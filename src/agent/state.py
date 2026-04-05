from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Conversation history
    messages: Annotated[list, add_messages]

    # Conversation flow
    phase: str              # "identity" | "incident" | "classified"
    intent: str             # "unknown" | "theft" | "other" | "handoff"

    # Client identity
    client_id: str
    policy_number: str
    client_name: str
    client_validated: bool
    validation_attempts: int

    # Incident details
    theft_confirmed: bool
    theft_date: str
    classification_attempts: int
