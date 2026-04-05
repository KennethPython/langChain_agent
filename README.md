# LangGraph Agent — Joule Fietsverzekering

A LangGraph orchestration agent for handling Joule insurance claims, starting with bicycle theft.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env` and fill in your API keys:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=langchain-agent
GROQ_API_KEY=your_key
```

## Project structure

```
src/agent/
  state.py          # AgentState definition
  db.py             # Mock client database lookups
  logger.py         # Business event logging to logs/events.json
  node1_chatbot.py  # Node 1: identity verification + intent classification
  router.py         # Conditional edge router
  graph.py          # Full graph wiring

data/
  mock_clients.json # 10 test clients (client_id + policy_number)

logs/
  events.json       # Auto-generated business event log
```

## Running

```bash
python main.py
```

## Testing with pytest

### Run all tests

```bash
python -m pytest tests/ -v
```

### Run a specific test file

```bash
python -m pytest tests/test_node1.py -v
```

### Run a specific test class

```bash
python -m pytest tests/test_node1.py::TestValidateClient -v
```

### Run a single test

```bash
python -m pytest tests/test_node1.py::TestNode1Chatbot::test_valid_identity_transitions_to_incident -v
```

### Run only tests that match a keyword

```bash
python -m pytest tests/ -k "theft" -v
```

### What is tested per node

| Test class | What it covers |
|---|---|
| `TestValidateClient` | DB lookups — valid and invalid credentials |
| `TestGetClientById` | DB lookup by client ID only |
| `TestExtractClientId` | Regex extraction of 6-digit client IDs from free text |
| `TestExtractPolicyNumber` | Regex extraction of POL-YEAR-NNNNN policy numbers |
| `TestLooksLikeTheft` | Keyword-based theft detection including negation |
| `TestNode1Chatbot` | Full node logic with mocked LLM — phase transitions, validation, routing triggers |
| `TestRouter` | Conditional edge routing — all four outcomes |

> The LLM (Groq) is always mocked in tests. No API calls are made during `pytest`.

## Mock client data

Test credentials for manual testing:

| Name | Client ID | Policy number |
|---|---|---|
| Jan De Smedt | 112233 | POL-2024-00101 |
| Marie Janssen | 224455 | POL-2023-00247 |
| Pieter Claes | 336677 | POL-2024-00389 |
| Sara Vermeersch | 448899 | POL-2022-00512 |
| Thomas Bogaert | 551122 | POL-2024-00634 |
