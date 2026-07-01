"""
Extraction Agent.

Job: pull structured entities out of raw policy text using spaCy NER plus a
domain EntityRuler for policy-specific terms, then validate/normalize them
through the Pydantic `Entity` schema.

If spaCy or its model isn't installed, it falls back to regex patterns so the
pipeline still returns entities.
"""
from __future__ import annotations

import logging
import re

from app.agents.base import BaseAgent
from app.core.config import get_settings
from app.models.schemas import Entity

logger = logging.getLogger(__name__)
_settings = get_settings()


# Domain terms the base NER model won't know about. Added via an EntityRuler
# so they surface as POLICY_TERM entities alongside the statistical NER.
_POLICY_TERMS = [
    "deductible", "coverage limit", "copay", "coinsurance", "premium",
    "out-of-network", "in-network", "effective date", "waiting period",
    "exclusion", "pre-existing condition", "claim", "appeal", "reimbursement",
]

_nlp = None


def _load_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        nlp = spacy.load(_settings.spacy_model)
        # Add a case-insensitive ruler for domain terms.
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(
            [{"label": "POLICY_TERM", "pattern": [{"LOWER": w} for w in term.split()]}
             for term in _POLICY_TERMS]
        )
        _nlp = nlp
    except Exception as exc:  # pragma: no cover - model dependent
        logger.warning("spaCy unavailable (%s); using regex fallback.", exc)
        _nlp = False
    return _nlp


_FALLBACK_PATTERNS = {
    "MONEY": re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    "PERCENT": re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    "DATE": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "DURATION": re.compile(r"\b\d+\s+days?\b", re.IGNORECASE),
}


def _normalize(label: str, text: str) -> str | None:
    """Normalize common value types so downstream logic can compare them."""
    if label in {"PERCENT"} or text.strip().endswith("%"):
        m = re.search(r"\d+(?:\.\d+)?", text)
        if m:
            return str(float(m.group()) / 100.0)
    if label in {"MONEY"}:
        m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
        if m:
            return m.group().replace(",", "")
    return None


class ExtractionAgent(BaseAgent):
    name = "extraction_agent"

    def run(self, text: str, source_doc: str) -> list[Entity]:
        nlp = _load_nlp()
        if nlp:
            return self._extract_spacy(nlp, text, source_doc)
        return self._extract_fallback(text, source_doc)

    def _extract_spacy(self, nlp, text: str, source_doc: str) -> list[Entity]:
        doc = nlp(text)
        entities: list[Entity] = []
        for ent in doc.ents:
            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    source_doc=source_doc,
                    normalized=_normalize(ent.label_, ent.text),
                )
            )
        return entities

    def _extract_fallback(self, text: str, source_doc: str) -> list[Entity]:
        entities: list[Entity] = []
        for label, pattern in _FALLBACK_PATTERNS.items():
            for m in pattern.finditer(text):
                entities.append(
                    Entity(
                        text=m.group(0),
                        label=label,
                        start_char=m.start(),
                        end_char=m.end(),
                        source_doc=source_doc,
                        normalized=_normalize(label, m.group(0)),
                    )
                )
        return entities
