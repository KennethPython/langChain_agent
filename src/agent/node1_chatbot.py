import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import AgentState
from .db import validate_client, get_client_by_id
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
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = set(cleaned.split())
    has_theft = bool(words & _THEFT_KEYWORDS)
    has_negation = bool(words & _NEGATION_WORDS)
    return has_theft and not has_negation


def _build_system_prompt(state: AgentState) -> str:
    base = (
        "Je naam is Suzy. Je bent een vriendelijke en empathische klantenservice medewerker van Joule, een fietslease service. "
        "Je communiceert uitsluitend in het Nederlands. "
        "Je toon is warm, begripvol en behulpzaam — als een vriend die toevallig expert is. "
        "Gebruik de voornaam van de klant zodra je die kent.\n\n"
    )

    phase = state.get("phase", "greeting")
    validation_attempts = state.get("validation_attempts", 0)
    classification_attempts = state.get("classification_attempts", 0)

    if phase == "greeting":
        instructions = (
            "HUIDIGE FASE: Begroeting\n"
            "Stel jezelf voor als Suzy van Joule. "
            "Verwelkom de klant hartelijk en vraag hoe je hen vandaag kan helpen.\n"
        )

    elif phase == "incident":
        if classification_attempts == 0:
            instructions = (
                "HUIDIGE FASE: Incidentbeschrijving\n"
                "Luister naar de klant. Toon begrip en vraag vriendelijk wat er precies is gebeurd "
                "als dat nog niet duidelijk is.\n"
            )
        else:
            instructions = (
                "HUIDIGE FASE: Verduidelijking\n"
                "Het is nog niet duidelijk wat er is gebeurd. "
                "Vraag vriendelijk maar direct wat er met de fiets is gebeurd.\n"
            )

    elif phase == "identity":
        client_name = state.get("client_name", "")
        has_id = bool(state.get("client_id", ""))
        has_policy = bool(state.get("policy_number", ""))

        if validation_attempts == 0 and not has_id and not has_policy:
            instructions = (
                "HUIDIGE FASE: Diefstal bevestigd — informeer en identificeer\n"
                "De klant heeft gemeld dat zijn/haar fiets gestolen is. Doe twee dingen in één bericht:\n"
                "1. Bevestig dat je begrijpt dat de fiets gestolen is en dat dit heel vervelend is.\n"
                "2. Informeer de klant dat zij EERST aangifte moeten doen bij de politie om een "
                "proces-verbaal (PV) te bekomen — dit is noodzakelijk voor de verdere afhandeling.\n"
                "3. Vraag vervolgens hun gegevens op om hen te identificeren:\n"
                "   - Klantnummer (6 cijfers)\n"
                "   - Polisnummer (formaat: POL-JAAR-NNNNN, bijv. POL-2024-00123)\n"
            )
        elif has_id and not has_policy:
            name_part = client_name.split()[0] if client_name else "daar"
            instructions = (
                f"HUIDIGE FASE: Klantnummer ontvangen, polisnummer ontbreekt\n"
                f"Je kent de klant nu als {client_name}. "
                f"Reageer kort en vriendelijk, gebruik de voornaam ({name_part}). "
                f"Vraag enkel naar het polisnummer (formaat: POL-JAAR-NNNNN). "
                f"Geen excuses, geen herhaling — gewoon kort en warm.\n"
            )
        else:
            remaining = MAX_VALIDATION_ATTEMPTS - validation_attempts
            instructions = (
                f"HUIDIGE FASE: Identiteitsverificatie — poging {validation_attempts + 1} van {MAX_VALIDATION_ATTEMPTS}\n"
                f"De gegevens kwamen niet overeen. "
                f"Vraag kort of er een typfout is en of de klant de gegevens opnieuw wil invoeren. "
                f"Er {'is nog 1 poging' if remaining == 1 else f'zijn nog {remaining} pogingen'} over.\n"
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
    phase = state.get("phase", "greeting")
    client_id = state.get("client_id", "")
    policy_number = state.get("policy_number", "")
    validation_attempts = state.get("validation_attempts", 0)
    classification_attempts = state.get("classification_attempts", 0)

    updates: dict = {}

    # Get last user message text
    last_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            last_user_text = msg.content
            break

    # --- PHASE: GREETING ---
    if phase == "greeting" and last_user_text:
        updates["phase"] = "incident"
        updates["classification_attempts"] = 0
        phase = "incident"

    # --- PHASE: INCIDENT ---
    if phase == "incident" and last_user_text:
        if _looks_like_theft(last_user_text):
            updates["theft_confirmed"] = True
            updates["intent"] = "theft"   # classify immediately on detection
            updates["phase"] = "identity" # then ask for credentials
            log_event("intent_classified", {"intent": "theft"})
        else:
            new_attempts = classification_attempts + 1
            updates["classification_attempts"] = new_attempts
            if new_attempts >= MAX_CLASSIFICATION_ATTEMPTS:
                updates["intent"] = "other"
                updates["phase"] = "classified"
                log_event("intent_classified", {
                    "intent": "other",
                    "reason": "max_classification_attempts_reached",
                })

    # --- PHASE: IDENTITY ---
    elif phase == "identity" and last_user_text:
        # Extract whatever is in this message
        extracted_id = _extract_client_id(last_user_text)
        extracted_policy = _extract_policy_number(last_user_text)

        # Accumulate: use newly extracted value or fall back to what's already in state
        current_id = extracted_id or client_id
        current_policy = extracted_policy or policy_number

        if extracted_id:
            updates["client_id"] = extracted_id
        if extracted_policy:
            updates["policy_number"] = extracted_policy

        # If we have an ID but no policy yet, look up the name so the bot can use it
        if current_id and not current_policy:
            known = get_client_by_id(current_id)
            if known and not state.get("client_name"):
                updates["client_name"] = known["client_name"]

        # Attempt validation as soon as both pieces are available
        if current_id and current_policy:
            client = validate_client(current_id, current_policy)
            if client:
                updates["client_validated"] = True
                updates["client_name"] = client["client_name"]
                updates["intent"] = "theft"
                updates["phase"] = "classified"
                log_event("identity_validated", {
                    "client_id": current_id,
                    "policy_number": current_policy,
                    "client_name": client["client_name"],
                    "intent": "theft",
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

    # --- Build prompt and call LLM ---
    merged_state = {**state, **updates}
    system_prompt = _build_system_prompt(merged_state)
    llm_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = _llm().invoke(llm_messages)

    return {**updates, "messages": [response]}
