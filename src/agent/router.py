from langgraph.graph import END
from .state import AgentState


def route_after_node1(state: AgentState) -> str:
    phase = state.get("phase", "identity")
    intent = state.get("intent", "unknown")

    if phase != "classified" and intent != "handoff":
        return END  # still in conversation, exit graph for this turn

    if intent == "theft":
        return "node2_documents"
    elif intent == "handoff":
        return "node_handoff"
    else:
        return "node_rag_joule"
