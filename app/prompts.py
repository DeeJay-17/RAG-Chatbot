from __future__ import annotations

from typing import Literal
from functools import lru_cache
import os
import yaml

PromptStyle = Literal["zero_shot", "one_shot", "few_shot", "chain_of_thought", "stepwise"]


def normalize_prompt_style(style: str | None) -> PromptStyle:
    if not style:
        return "zero_shot"
    s = style.strip().lower()
    if s in ("zero_shot", "zero-shot"):
        return "zero_shot"
    if s in ("one_shot", "one-shot"):
        return "one_shot"
    if s in ("few_shot", "few-shot"):
        return "few_shot"
    if s in ("chain_of_thought", "cot", "chain-of-thought"):
        return "chain_of_thought"
    if s in ("stepwise", "step_by_step", "step-by-step"):
        return "stepwise"
    return "zero_shot"


@lru_cache(maxsize=1)
def _load_prompts_yaml() -> dict:
    """
    Load and cache the central prompts.yaml file.
    """
    here = os.path.dirname(__file__)
    path = os.path.join(here, "prompts.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pdf_answer_system_prompt(style: str | None) -> str:
    """
    System prompt for answering questions from a single chosen PDF chunk.

    Backed by the pdf_answer section of prompts.yaml.
    """
    s = normalize_prompt_style(style)
    data = _load_prompts_yaml()["pdf_answer"]
    base: str = data["base"]
    extra: str = data["styles"][s]
    return base + "\n\n" + extra


def csv_postprocess_system_prompt(style: str | None) -> str:
    """
    System prompt for turning SQL result rows into a natural language answer.

    Backed by the csv_postprocess section of prompts.yaml.
    """
    s = normalize_prompt_style(style)
    data = _load_prompts_yaml()["csv_postprocess"]
    base: str = data["base"]
    extra: str = data["styles"][s]
    return base + "\n\n" + extra


def chat_route_system_prompt(file_context: str) -> str:
    """
    System prompt for the /chat/route router, with dynamic file context injected.
    """
    tmpl: str = _load_prompts_yaml()["chat_route"]["system_template"]
    return tmpl.format(file_context=file_context)


def chat_csv_sql_system_prompt() -> str:
    """
    System prompt for /chat/csv_sql SQL generation over a single CSV table.
    """
    return _load_prompts_yaml()["chat_csv_sql"]["system"]


def chat_answer_csv_sql_system_prompt() -> str:
    """
    System prompt for multi-table SQL generation inside /chat/answer.
    """
    return _load_prompts_yaml()["chat_answer"]["csv_sql_system"]
