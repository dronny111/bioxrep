from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class NotationForm:
    text: str
    notation: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class EquivalenceClass:
    fact_id: str
    track: str
    forms: List[NotationForm]
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "track": self.track,
            "forms": [form.to_dict() for form in self.forms],
            "attributes": self.attributes,
        }
