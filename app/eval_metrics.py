# ============================================
# FILE: app/eval_metrics.py
# ============================================
from typing import List, Dict, Any, Optional
import re

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def recall_at_k(retrieved: List[Dict[str, Any]], relevant_predicate, k: int) -> float:
    """
    retrieved: list of items (each contains at least 'text' or 'id')
    relevant_predicate: function(item)->bool
    """
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return 1.0 if any(relevant_predicate(x) for x in topk) else 0.0

def mrr_at_k(retrieved: List[Dict[str, Any]], relevant_predicate, k: int) -> float:
    topk = retrieved[:k]
    for i, item in enumerate(topk, start=1):
        if relevant_predicate(item):
            return 1.0 / i
    return 0.0
