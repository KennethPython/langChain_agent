import json
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "mock_clients.json"
_clients: list[dict] | None = None


def _load() -> list[dict]:
    global _clients
    if _clients is None:
        with open(_DB_PATH, encoding="utf-8") as f:
            _clients = json.load(f)
    return _clients


def validate_client(client_id: str, policy_number: str) -> dict | None:
    """Return client record if client_id and policy_number match, else None."""
    for client in _load():
        if client["client_id"] == client_id and client["policy_number"] == policy_number:
            return client
    return None


def get_client_by_id(client_id: str) -> dict | None:
    for client in _load():
        if client["client_id"] == client_id:
            return client
    return None
