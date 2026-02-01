# app/pii.py
from __future__ import annotations

from typing import Any, Dict, List

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


_analyzer = AnalyzerEngine()  # Uses default spaCy-based NLP engine
_anonymizer = AnonymizerEngine()


def mask_text(text: str | None, language: str = "en") -> str | None:

    if not text:
        return text

    results = _analyzer.analyze(
        text=text,
        entities=[],  # empty means "use all supported PII entities"
        language=language,
    )
    if not results:
        return text

    anonymized = _anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )
    return anonymized.text


def mask_rows(rows: List[Dict[str, Any]], language: str = "en") -> List[Dict[str, Any]]:

    masked: List[Dict[str, Any]] = []
    for row in rows:
        new_row: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str):
                new_row[k] = mask_text(v, language=language)
            else:
                new_row[k] = v
        masked.append(new_row)
    return masked

