"""Tests for node 1: identity validation, intent extraction, and routing."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.db import validate_client, get_client_by_id
from src.agent.node1_chatbot import (
    _extract_client_id,
    _extract_policy_number,
    _looks_like_theft,
    node1_chatbot,
)
from src.agent.router import route_after_node1
from src.agent.state import AgentState


# ---------------------------------------------------------------------------
# db.py tests
# ---------------------------------------------------------------------------

class TestValidateClient:
    def test_valid_credentials(self):
        result = validate_client("112233", "POL-2024-00101")
        assert result is not None
        assert result["client_name"] == "Jan De Smedt"

    def test_wrong_client_id(self):
        assert validate_client("000000", "POL-2024-00101") is None

    def test_wrong_policy_number(self):
        assert validate_client("112233", "POL-2024-99999") is None

    def test_both_wrong(self):
        assert validate_client("000000", "POL-0000-00000") is None

    def test_all_clients_loadable(self):
        # All 10 mock clients should validate against their own credentials
        from src.agent.db import _load
        for client in _load():
            result = validate_client(client["client_id"], client["policy_number"])
            assert result is not None, f"Failed for {client['client_id']}"


class TestGetClientById:
    def test_existing_client(self):
        result = get_client_by_id("224455")
        assert result["client_name"] == "Marie Janssen"

    def test_nonexistent_client(self):
        assert get_client_by_id("999999") is None


# ---------------------------------------------------------------------------
# Extraction helper tests
# ---------------------------------------------------------------------------

class TestExtractClientId:
    def test_plain_number(self):
        assert _extract_client_id("mijn nummer is 112233") == "112233"

    def test_number_in_sentence(self):
        assert _extract_client_id("klantnummer 336677 alsjeblieft") == "336677"

    def test_no_match(self):
        assert _extract_client_id("ik weet het niet") is None

    def test_ignores_short_numbers(self):
        assert _extract_client_id("12345") is None  # only 5 digits

    def test_ignores_long_numbers(self):
        assert _extract_client_id("1234567") is None  # 7 digits, not standalone 6


class TestExtractPolicyNumber:
    def test_uppercase(self):
        assert _extract_policy_number("POL-2024-00101") == "POL-2024-00101"

    def test_lowercase_input(self):
        assert _extract_policy_number("pol-2024-00101") == "POL-2024-00101"

    def test_in_sentence(self):
        assert _extract_policy_number("mijn polis is POL-2023-00247 dankuwel") == "POL-2023-00247"

    def test_no_match(self):
        assert _extract_policy_number("geen polis hier") is None


class TestLooksLikeTheft:
    def test_gestolen(self):
        assert _looks_like_theft("mijn fiets is gestolen") is True

    def test_kwijt(self):
        assert _looks_like_theft("ik ben mijn fiets kwijt") is True

    def test_weg(self):
        assert _looks_like_theft("mijn fiets is weg") is True

    def test_negation(self):
        assert _looks_like_theft("mijn fiets is niet gestolen") is False

    def test_unrelated(self):
        assert _looks_like_theft("ik wil informatie over mijn verzekering") is False


# ---------------------------------------------------------------------------
# node1_chatbot tests (LLM mocked)
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "messages": [],
        "phase": "identity",
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
    return {**defaults, **kwargs}


@pytest.fixture
def mock_llm_response():
    """Fixture that mocks ChatGroq to return a fixed AI message."""
    fake_response = AIMessage(content="Dit is een testantwoord van de bot.")
    with patch("src.agent.node1_chatbot.ChatGroq") as mock_cls:
        instance = MagicMock()
        instance.invoke.return_value = fake_response
        mock_cls.return_value = instance
        yield instance


class TestNode1Chatbot:
    def test_initial_greeting_phase_identity(self, mock_llm_response):
        state = _make_state(messages=[HumanMessage(content="hallo")])
        result = node1_chatbot(state)
        assert "messages" in result
        assert mock_llm_response.invoke.called

    def test_valid_identity_transitions_to_incident(self, mock_llm_response):
        state = _make_state(messages=[
            HumanMessage(content="klantnummer 112233 polis POL-2024-00101")
        ])
        result = node1_chatbot(state)
        assert result.get("phase") == "incident"
        assert result.get("client_validated") is True
        assert result.get("client_name") == "Jan De Smedt"

    def test_invalid_identity_increments_attempts(self, mock_llm_response):
        state = _make_state(messages=[
            HumanMessage(content="klantnummer 000000 polis POL-0000-00000")
        ])
        result = node1_chatbot(state)
        assert result.get("validation_attempts") == 1
        # phase is not explicitly returned when unchanged — stays "identity"
        assert result.get("phase", "identity") == "identity"

    def test_three_failed_attempts_triggers_handoff(self, mock_llm_response):
        state = _make_state(
            validation_attempts=2,
            messages=[HumanMessage(content="klantnummer 000000 polis POL-0000-00000")]
        )
        result = node1_chatbot(state)
        assert result.get("intent") == "handoff"
        assert result.get("phase") == "handoff"

    def test_theft_detected_in_incident_phase(self, mock_llm_response):
        state = _make_state(
            phase="incident",
            client_validated=True,
            client_name="Jan De Smedt",
            messages=[HumanMessage(content="mijn fiets is gestolen gisteren avond")]
        )
        result = node1_chatbot(state)
        assert result.get("theft_confirmed") is True
        assert result.get("phase") == "theft_date"

    def test_theft_date_sets_classified(self, mock_llm_response):
        state = _make_state(
            phase="theft_date",
            client_validated=True,
            theft_confirmed=True,
            messages=[HumanMessage(content="gisteren rond 22u")]
        )
        result = node1_chatbot(state)
        assert result.get("intent") == "theft"
        assert result.get("phase") == "classified"
        assert result.get("theft_date") == "gisteren rond 22u"

    def test_max_classification_attempts_routes_to_other(self, mock_llm_response):
        state = _make_state(
            phase="incident",
            classification_attempts=2,
            client_validated=True,
            messages=[HumanMessage(content="ik heb een vraag over mijn contract")]
        )
        result = node1_chatbot(state)
        assert result.get("intent") == "other"
        assert result.get("phase") == "classified"


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestRouter:
    def test_theft_routes_to_node2(self):
        state = _make_state(phase="classified", intent="theft")
        assert route_after_node1(state) == "node2_documents"

    def test_other_routes_to_rag_joule(self):
        state = _make_state(phase="classified", intent="other")
        assert route_after_node1(state) == "node_rag_joule"

    def test_handoff_routes_to_handoff(self):
        state = _make_state(phase="handoff", intent="handoff")
        assert route_after_node1(state) == "node_handoff"

    def test_still_in_conversation_returns_end(self):
        from langgraph.graph import END
        state = _make_state(phase="identity", intent="unknown")
        assert route_after_node1(state) == END
