from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .node1_chatbot import node1_chatbot
from .router import route_after_node1


def _placeholder(state: AgentState) -> dict:
    return {}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("node1_chatbot", node1_chatbot)
    graph.add_node("node2_documents", _placeholder)   # TODO: node 2
    graph.add_node("node_rag_joule", _placeholder)    # TODO: rag_joule node
    graph.add_node("node_handoff", _placeholder)      # TODO: handoff node

    graph.add_edge(START, "node1_chatbot")
    graph.add_conditional_edges(
        "node1_chatbot",
        route_after_node1,
        {
            END: END,
            "node2_documents": "node2_documents",
            "node_rag_joule": "node_rag_joule",
            "node_handoff": "node_handoff",
        },
    )

    graph.add_edge("node2_documents", END)
    graph.add_edge("node_rag_joule", END)
    graph.add_edge("node_handoff", END)

    return graph.compile()
