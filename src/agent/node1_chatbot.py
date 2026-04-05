import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import AgentState
from .db import validate_client
from .logger import log_event

MAX_VALIDATION_ATTEMPTS = 3
MAX_CLASSIFICATION_ATTEMPTS = 3

_THEFT_KEYWORDS = {"gestolen", "diefstal", "weg", "verdwenen", "kwijt", "gejat", "stelen"}
_NEGATION_WORDS = {"niet", "nee", "geen", "nooit"}


def _llm() -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


def _extract_client_id(text: str) -> str | None:
    match = re.search(r"\b(\d{6})\b", text)
    return match.group(1) if match else None


def _extract_policy_number(text: str) -> str | None:
    match = re.search(r"(POL-\d{4}-\d{5})", text.upper())
    return match.group(1) if match else None


def _looks_like_theft(text: str) -> bool:
    words = set(text.lower().split())
    has_theft = bool(words & _THEFT_KEYWORDS)
    has_negation = bool(words & _NEGATION_WORDS)
    return has_theft and not has_negation


def _build_system_prompt(state: AgentState) -> str:
    base = (
        "Je bent een vriendelijke en empathische klantenservice medewerker van Joule Fietsverzekering. "
        "Je communiceert uitsluitend in het Nederlands. "
        "Je toon is warm, begripvol en behulpzaam — als een vriend die toevallig expert is. "
        "Gebruik de voornaam van de klant zodra je die kent.\n\n"
    )

    phase = state.get("phase", "identity")
    attempts = state.get("validation_attempts", 0)
    client_name = state.get("client_name", "")
    classification_attempts = state.get("classification_attempts", 0)

    if phase == "identity":
        if attempts == 0:
            instructions = (
                "HUIDIGE FASE: Identiteitsverificatie\n"
                "Verwelkom de klant hartelijk. Leg uit dat je hen graag wil helpen, "
                "maar dat je eerst hun gegevens nodig hebt om hen te identificeren.\n"
                "Vraag naar:\n"
                "- Klantnummer (6 cijfers)\n"
                "- Polisnummer (formaat: POL-JAAR-NNNNN, bijv. POL-2024-00123)\n"
            )
        else:
            remaining = MAX_VALIDATION_ATTEMPTS - attempts
            instructions = (
                f"HUIDIGE FASE: Identiteitsverificatie — poging {attempts + 1} van {MAX_VALIDATION_ATTEMPTS}\n"
                f"De gegevens kwamen {attempts} keer niet overeen. "
                f"Vraag vriendelijk of er misschien een typfout is en of de klant de gegevens opnieuw wil invoeren. "
                f"Er {'is nog 1 poging' if remaining == 1 else f'zijn nog {remaining} pogingen'} over.\n"
            )

    elif phase == "incident":
        greeting = f"De klant heet {client_name}. " if client_name else ""
        if classification_attempts == 0:
            instructions = (
                f"HUIDIGE FASE: Incidentbeschrijving\n"
                f"{greeting}"
                "Vraag de klant vriendelijk wat er aan de hand is en toon begrip voor hun situatie.\n"
            )
        elif classification_attempts == 1:
            instructions = (
                "HUIDIGE FASE: Verduidelijking\n"
                f"{greeting}"
                "Het is nog niet helemaal duidelijk wat er is gebeurd. "
                "Vraag expliciet en direct: 'Is uw fiets gestolen?'\n"
            )
        else:
            instructions = (
                "HUIDIGE FASE: Bevestiging\n"
                f"{greeting}"
                "Vraag de klant om iets meer details over wat er precies is gebeurd, "
                "zodat je hen zo goed mogelijk kan helpen.\n"
            )

    elif phase == "theft_date":
        instructions = (
            "HUIDIGE FASE: Datum van diefstal\n"
            f"De klant heeft bevestigd dat de fiets gestolen is. "
            "Vraag wanneer de diefstal heeft plaatsgevonden (datum of tijdstip).\n"
        )

    elif phase == "handoff":
        instructions = (
            "HUIDIGE FASE: Doorverwijzing naar medewerker\n"
            "Informeer de klant vriendelijk maar duidelijk dat er een probleem is met de identiteitsverificatie. "
            "Zeg dat er een incidentmelding is aangemaakt en dat een medewerker van Joule "
            "hen binnen 4 uur zal contacteren. Wens de klant een fijne dag.\n"
        )

    else:
        instructions = ""

    return base + instructions


def node1_chatbot(state: AgentState) -> dict:
    messages = state.get("messages", [])
    phase = state.get("phase", "identity")
    intent = state.get("intent", "unknown")
    client_id = state.get("client_id", "")
    policy_number = state.get("policy_number", "")
    client_validated = state.get("client_validated", False)
    validation_attempts = state.get("validation_attempts", 0)
    classification_attempts = state.get("classification_attempts", 0)
    theft_confirmed = state.get("theft_confirmed", False)

    updates: dict = {}

    # Get last user message text
    last_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            last_user_text = msg.content if hasattr(msg, "content") else msg.get("content", "")
            break

    # --- PHASE: IDENTITY ---
    if phase == "identity":
        extracted_id = _extract_client_id(last_user_text)
        extracted_policy = _extract_policy_number(last_user_text)

        if extracted_id:
            updates["client_id"] = extracted_id
            client_id = extracted_id
        if extracted_policy:
            updates["policy_number"] = extracted_policy
            policy_number = extracted_policy

        current_id = updates.get("client_id", client_id)
        current_policy = updates.get("policy_number", policy_number)

        if current_id and current_policy:
            client = validate_client(current_id, current_policy)
            if client:
                updates["client_validated"] = True
                updates["client_name"] = client["client_name"]
                updates["phase"] = "incident"
                updates["classification_attempts"] = 0
                log_event("identity_validated", {
                    "client_id": current_id,
                    "policy_number": current_policy,
                    "client_name": client["client_name"],
                })
            else:
                new_attempts = validation_attempts + 1
                updates["validation_attempts"] = new_attempts
                if new_attempts >= MAX_VALIDATION_ATTEMPTS:
                    updates["phase"] = "handoff"
                    updates["intent"] = "handoff"
                    log_event("identity_validation_failed", {
                        "client_id": current_id,
                        "policy_number": current_policy,
                        "attempts": new_attempts,
                        "action": "handoff",
                    })

    # --- PHASE: INCIDENT ---
    elif phase == "incident":
        if _looks_like_theft(last_user_text):
            updates["theft_confirmed"] = True
            updates["phase"] = "theft_date"
        else:
            new_attempts = classification_attempts + 1
            updates["classification_attempts"] = new_attempts

            if new_attempts >= MAX_CLASSIFICATION_ATTEMPTS:
                # Can't classify — route to rag_joule
                updates["intent"] = "other"
                updates["phase"] = "classified"
                log_event("intent_classified", {
                    "client_id": client_id,
                    "policy_number": policy_number,
                    "intent": "other",
                    "reason": "max_classification_attempts_reached",
                })

    # --- PHASE: THEFT_DATE ---
    elif phase == "theft_date":
        # Whatever the client says is the date/context — accept it
        updates["theft_date"] = last_user_text.strip()
        updates["intent"] = "theft"
        updates["phase"] = "classified"
        log_event("intent_classified", {
            "client_id": client_id,
            "policy_number": policy_number,
            "intent": "theft",
            "theft_date": last_user_text.strip(),
        })

    # --- Build prompt and call LLM ---
    merged_state = {**state, **updates}
    system_prompt = _build_system_prompt(merged_state)

    llm_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = _llm().invoke(llm_messages)

    return {**updates, "messages": [response]}
