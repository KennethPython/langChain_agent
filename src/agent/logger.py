import json
from datetime import datetime
from pathlib import Path

_LOG_FILE = Path(__file__).parent.parent.parent / "logs" / "events.json"


def log_event(event_type: str, data: dict) -> None:
    """Append a structured business event to logs/events.json."""
    _LOG_FILE.parent.mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        **data,
    }

    events: list[dict] = []
    if _LOG_FILE.exists():
        with open(_LOG_FILE, encoding="utf-8") as f:
            try:
                events = json.load(f)
            except json.JSONDecodeError:
                events = []

    events.append(entry)

    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
