# LangGraph Agent — Joule Fietslease

A LangGraph orchestration agent for handling Joule insurance claims, starting with bicycle theft.

## Business flow

```
Client start gesprek
  └─ Node 1: Chatbot
       ├─ Begroeting door Suzy
       ├─ Classificatie: diefstal of andere vraag
       │    ├─ Diefstal → informeer over PV → identiteitsverificatie
       │    │    ├─ Geverifieerd → THEFT → Node 2 (documenten)
       │    │    └─ 3x mislukt → HANDOFF → medewerker neemt contact op
       │    └─ Andere vraag → RAG Joule (algemene vragen)
```

**Nog te bouwen:**
- Node 2: document upload (PV, sleutelfoto, eigendomsdocument)
- Node 3: diefstal dossier aanmaken (THEFT-XXXX)
- Node 4: RAG polisopzoeking per klant
- Node 5: procedure & voorwaarden op basis van polis
- Node 6: feedback samenvatting aan klant

## Setup

```bash
pip install -r requirements.txt
```

Maak een `.env` bestand aan:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=langchain-agent
GROQ_API_KEY=your_key
```

## Starten

### Streamlit app

```bash
python -m streamlit run app.py
```

### CLI

```bash
python main.py
```

## Project structuur

```
src/agent/
  state.py          # AgentState definitie
  db.py             # Mock client database (10 testklanten)
  logger.py         # Business event logging → logs/events.json
  node1_chatbot.py  # Node 1: begroeting, classificatie, identiteitsverificatie
  router.py         # Conditional edge router
  graph.py          # Volledige graph

app.py              # Streamlit web interface
data/
  mock_clients.json # 10 testklanten
logs/
  events.json       # Automatisch gegenereerd event log
```

## Testen met pytest

```bash
# Alle tests
python -m pytest tests/ -v

# Specifiek bestand
python -m pytest tests/test_node1.py -v

# Specifieke klasse
python -m pytest tests/test_node1.py::TestValidateClient -v

# Keyword filter
python -m pytest tests/ -k "theft" -v
```

### Wat wordt getest

| Test klasse | Wat het dekt |
|---|---|
| `TestValidateClient` | DB lookups — geldige en ongeldige credentials |
| `TestGetClientById` | DB lookup op klantnummer alleen |
| `TestExtractClientId` | Regex extractie 6-cijferig klantnummer uit vrije tekst |
| `TestExtractPolicyNumber` | Regex extractie POL-JAAR-NNNNN polisnummer |
| `TestLooksLikeTheft` | Diefstal detectie inclusief negaties |
| `TestNode1Chatbot` | Volledige node logica met gemockte LLM |
| `TestRouter` | Conditional edge routing — alle uitkomsten |

> De LLM (Groq) wordt altijd gemockt tijdens tests. Geen API calls.

## Testklanten

| Naam | Klantnummer | Polisnummer |
|---|---|---|
| Jan De Smedt | 112233 | POL-2024-00101 |
| Marie Janssen | 224455 | POL-2023-00247 |
| Pieter Claes | 336677 | POL-2024-00389 |
| Sara Vermeersch | 448899 | POL-2022-00512 |
| Thomas Bogaert | 551122 | POL-2024-00634 |
| Elien Vandenberghe | 663344 | POL-2023-00756 |
| Kevin Martens | 775566 | POL-2024-00878 |
| Nathalie Leclercq | 887788 | POL-2022-00990 |
| Bram Wouters | 990011 | POL-2023-01012 |
| Annelies Peeters | 101010 | POL-2024-01134 |
