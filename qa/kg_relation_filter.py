from typing import Iterable, List, Optional, Set, Tuple


def normalize_relationships(
    rows: Iterable[Tuple[str, str, str]],
    allowed_types: Optional[Set[str]],
    max_items: int,
) -> List[str]:
    if max_items <= 0:
        return []

    normalized: List[str] = []
    seen = set()

    for start_name, rel_type, end_name in rows:
        if allowed_types and rel_type not in allowed_types:
            continue
        value = f"{start_name} {rel_type} {end_name}"
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= max_items:
            break

    return normalized
