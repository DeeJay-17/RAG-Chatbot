# FILE: app/llm_client.py
import os
from dotenv import load_dotenv

load_dotenv()

from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


class LLMError(Exception):
    pass


def generate_text(prompt: str, model: str | None = None) -> str:
    """
    Simple Gemini text generation.
    """
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=model or GEMINI_MODEL,
        contents=prompt
    )

    text = getattr(resp, "text", None)
    if not text:
        raise LLMError(f"Empty response from Gemini: {resp}")
    return text


def chat_completion(messages: list[dict], temperature: float = 0.0) -> str:
    """
    Drop-in replacement for prior OpenAI-style function.

    We convert role-based messages into a single prompt because
    `generate_content` is prompt-driven in your usage.
    """
    # Combine into one prompt
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"{role}:\n{content}\n")

    prompt = "\n".join(parts).strip()
    return generate_text(prompt)



# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2-uncensored:7b")


# class LLMError(Exception):
#     pass


# def generate_text(prompt: str, model: str | None = None) -> str:
#     """
#     Simple Ollama text generation.
#     """
#     url = f"{OLLAMA_BASE_URL}/api/generate"
#     payload = {
#         "model": model or OLLAMA_MODEL,
#         "prompt": prompt,
#         "stream": False,
#     }

#     try:
#         resp = requests.post(url, json=payload, timeout=120)
#         resp.raise_for_status()
#     except requests.RequestException as e:
#         raise LLMError(f"Ollama request failed: {e}")

#     data = resp.json()
#     text = data.get("response")

#     if not text:
#         raise LLMError(f"Empty response from Ollama: {data}")

#     return text.strip()


# def chat_completion(messages: list[dict], temperature: float = 0.0) -> str:
#     """
#     OpenAI-style chat interface backed by Ollama.
#     """
#     parts = []
#     for m in messages:
#         role = (m.get("role") or "user").upper()
#         content = m.get("content") or ""
#         parts.append(f"{role}:\n{content}\n")

#     prompt = "\n".join(parts).strip()
#     return generate_text(prompt)
