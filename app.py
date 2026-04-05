import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.node1_chatbot import node1_chatbot
from src.agent.state import AgentState


def init_state() -> AgentState:
    return {
        "messages": [],
        "phase": "greeting",
        "intent": "unknown",
        "client_id": "",
        "policy_number": "",
        "client_name": "",
        "client_validated": False,
        "validation_attempts": 0,
        "theft_confirmed": False,
        "theft_date": "",
        "classification_attempts": 0,
    }


def run_node1(state: AgentState, user_text: str) -> AgentState:
    updated_messages = list(state["messages"]) + [HumanMessage(content=user_text)]
    result = node1_chatbot({**state, "messages": updated_messages})
    new_state = {**state, **result}
    new_state["messages"] = updated_messages + [
        m for m in result.get("messages", [])
        if isinstance(m, AIMessage)
    ]
    return new_state


# --- Page config ---
st.set_page_config(page_title="Joule Fietsverzekering", page_icon="🚲", layout="centered")
st.title("Joule Fietsverzekering")
st.caption("Klantenservice")

# --- Session state init ---
if "agent_state" not in st.session_state:
    st.session_state.agent_state = init_state()
    # Trigger opening message from bot
    opening = node1_chatbot({**st.session_state.agent_state, "messages": [HumanMessage(content="")]})
    st.session_state.agent_state = {**st.session_state.agent_state, **opening}
    st.session_state.agent_state["messages"] = [
        m for m in opening.get("messages", []) if isinstance(m, AIMessage)
    ]

state: AgentState = st.session_state.agent_state

# --- Status sidebar ---
with st.sidebar:
    st.subheader("Status")

    phase = state.get("phase", "identity")
    intent = state.get("intent", "unknown")

    phase_labels = {
        "greeting": "👋 Begroeting",
        "incident": "💬 Incidentbeschrijving",
        "theft_date": "📅 Datum diefstal",
        "identity": "🔐 Identiteitsverificatie",
        "classified": "✅ Geclassificeerd",
        "handoff": "🔁 Doorverwezen naar medewerker",
    }
    st.write("**Fase:**", phase_labels.get(phase, phase))

    if state.get("client_validated"):
        st.write("**Klant:**", state.get("client_name", ""))
        st.write("**Klantnummer:**", state.get("client_id", ""))
        st.write("**Polisnummer:**", state.get("policy_number", ""))

    if phase == "classified":
        if intent == "theft":
            st.success("Geclassificeerd als: **DIEFSTAL**")
            if state.get("theft_date"):
                st.write("**Datum:**", state.get("theft_date"))
        elif intent == "other":
            st.info("Geclassificeerd als: **ANDERE VRAAG** → doorgestuurd naar Joule RAG")
    elif phase == "handoff":
        st.error("Identiteitsverificatie mislukt — medewerker wordt gecontacteerd")

    st.divider()
    if st.button("Gesprek herstarten"):
        del st.session_state.agent_state
        st.rerun()

# --- Chat display ---
for msg in state.get("messages", []):
    if isinstance(msg, HumanMessage) and msg.content:
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🚲"):
            st.write(msg.content)

# --- Input ---
is_done = phase in ("classified", "handoff")

if is_done:
    st.info("Dit gesprek is afgerond. Gebruik 'Gesprek herstarten' in de zijbalk om opnieuw te beginnen.")
else:
    user_input = st.chat_input("Typ uw bericht...")
    if user_input:
        st.session_state.agent_state = run_node1(state, user_input)
        st.rerun()
