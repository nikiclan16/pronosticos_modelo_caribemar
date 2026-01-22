from typing import Iterable, Tuple, Dict, Any


def build_in_clause(values: Iterable[Any], prefix: str) -> Tuple[str, Dict[str, Any]]:
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")
    placeholders = []
    params: Dict[str, Any] = {}
    for idx, value in enumerate(values):
        key = f"{prefix}_{idx}"
        placeholders.append(f"%({key})s")
        params[key] = value
    return f"({', '.join(placeholders)})", params
