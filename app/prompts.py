from __future__ import annotations

"""
Prompt templates for different prompting techniques.

Supported styles (keys passed from frontend / API):
- "zero_shot"
- "one_shot"
- "few_shot"
- "chain_of_thought"
- "stepwise"
"""

from typing import Literal

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


def pdf_answer_system_prompt(style: str | None) -> str:
    """
    System prompt for answering questions from a single chosen PDF chunk.
    """
    s = normalize_prompt_style(style)

    base = (
        "You are a helpful assistant answering questions using a retrieved PDF chunk.\n"
        "Rules:\n"
        "- Use ONLY the provided chunk as evidence.\n"
        "- If the chunk does not contain the answer, say what is missing.\n"
        "- Do NOT omit any relevant facts from the chunk that help answer the question.\n"
        "- Do NOT summarize or paraphrase the wording of relevant sentences; instead, quote or closely mirror the original text.\n"
        "- You may add brief connective phrases (e.g., \"In summary,\" or \"This means that\") but do not shorten, merge, or rephrase the factual statements themselves.\n"
        "Always return STRICT JSON only: {\"answer\":\"...\"}.\n"
    )

    if s == "zero_shot":
        extra = "Use the chunk directly to answer; do not invent information beyond what is stated."
    elif s == "one_shot":
        extra = (
            "Here is an example of how to answer:\n"
            "Example user question: \"What is the primary benefit mentioned?\"\n"
            "Example chunk: \"The primary benefit of the system is improved scalability and easier maintenance.\"\n"
            "Example answer: \"The primary benefit mentioned is improved scalability and easier maintenance.\"\n"
            "Follow this style for the current question."
        )
    elif s == "few_shot":
        extra = (
            "Examples:\n"
            "1) Question: \"Who is the target audience?\"\n"
            "   Chunk: \"This document is written for data engineers and analytics teams.\"\n"
            "   Answer: \"The target audience is data engineers and analytics teams.\"\n"
            "2) Question: \"What two challenges are highlighted?\"\n"
            "   Chunk: \"The report highlights scalability and cost inefficiency as the main challenges.\"\n"
            "   Answer: \"The two highlighted challenges are scalability and cost inefficiency.\"\n"
            "Use similar concise answers for the current question."
        )
    elif s == "chain_of_thought":
        extra = (
            "Before producing the final answer, reason step by step in your head (do NOT show the reasoning), "
            "then output only the final concise answer in the JSON 'answer' field."
        )
    else:  # stepwise
        extra = (
            "Think through the problem in a few internal steps (identify relevant sentences, interpret them, "
            "then synthesize the conclusion), but only output the final concise answer in the JSON 'answer' field."
        )

    return base + "\n\n" + extra


def csv_postprocess_system_prompt(style: str | None) -> str:
    """
    System prompt for turning SQL result rows into a natural language answer.
    """
    s = normalize_prompt_style(style)

    base = (
        "You turn SQL query results into a concise user-facing answer.\n"
        "Rules:\n"
        "- Do NOT show a raw table.\n"
        "- Summarize key numbers, trends, and findings.\n"
        "- Be specific: mention values, counts, and categories when helpful.\n"
        "- If rows are empty, explain that no matching data was found.\n"
        "Always return STRICT JSON only: {\"answer\":\"...\"}.\n"
    )

    if s == "zero_shot":
        extra = "Directly describe what the results say, in a short paragraph or a few bullet points."
    elif s == "one_shot":
        extra = (
            "Example:\n"
            "Rows: [{\"department\":\"Sales\",\"total_revenue\":120000},{\"department\":\"Marketing\",\"total_revenue\":80000}]\n"
            "Answer: \"Sales generated 120,000 in revenue, while Marketing generated 80,000.\"\n"
            "Follow this style for the current results."
        )
    elif s == "few_shot":
        extra = (
            "Examples:\n"
            "1) Rows: [{\"city\":\"A\",\"count\":10},{\"city\":\"B\",\"count\":5}]\n"
            "   Answer: \"City A has 10 records, while City B has 5.\"\n"
            "2) Rows: []\n"
            "   Answer: \"There are no records matching the requested filters.\"\n"
            "Use similarly short explanations for the current results."
        )
    elif s == "chain_of_thought":
        extra = (
            "First, analyze the rows and infer the most important insights internally. "
            "Then output only the final concise explanation in the JSON 'answer' field (do not show your reasoning)."
        )
    else:  # stepwise
        extra = (
            "Internally go through three steps (1. scan results, 2. identify main patterns, 3. summarize), "
            "then output only the final concise answer in the JSON 'answer' field."
        )

    return base + "\n\n" + extra

