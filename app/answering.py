"""Grounded answer generation with a safe no-API-key fallback."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Keep this in one place: model availability can differ by API project and tier.
MODEL_NAME = "gemini-3.7-flash"


def create_answer(question: str, sources: list[dict[str, Any]]) -> tuple[str, str, str | None]:
    """Answer only from retrieved evidence; never send the full PDF to the model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "I found relevant evidence below. Add a Gemini API key to generate a concise answer grounded in these sources.",
            "retrieval-only",
            "AI answer generation is off because GEMINI_API_KEY is not configured.",
        )

    prompt = build_prompt(question, sources)
    try:
        answer = call_gemini(api_key, prompt)
    except HTTPError as error:
        return (
            "I found relevant evidence below, but the AI answer generator is temporarily unavailable.",
            "retrieval-only",
            f"Gemini API returned HTTP {error.code}. Your key may be invalid, revoked, or unavailable for this model.",
        )
    except (URLError, TimeoutError):
        return (
            "I found relevant evidence below, but the AI answer generator is temporarily unavailable.",
            "retrieval-only",
            "CiteWise could not reach the Gemini API. Check your internet connection and try again.",
        )
    except (ValueError, KeyError):
        return (
            "I found relevant evidence below, but the AI answer generator is temporarily unavailable.",
            "retrieval-only",
            "Gemini returned an unexpected response. Showing source passages only.",
        )
    return answer, MODEL_NAME, None


def build_prompt(question: str, sources: list[dict[str, Any]]) -> str:
    evidence = "\n\n".join(
        f"[{number}] {source['document']}, page {source['page']}:\n{source['excerpt']}"
        for number, source in enumerate(sources, start=1)
    )
    return f"""You are CiteWise, a careful document assistant.
Answer the user's question using ONLY the evidence below.
If the evidence does not answer the question, say exactly: "I don't know based on the uploaded documents."
Do not use outside knowledge. Do not invent details. Give a short, clear answer in at most 120 words.
End each factual sentence with the evidence citation in square brackets, such as [1].

Question: {question}

Evidence:
{evidence}
"""


def call_gemini(api_key: str, prompt: str) -> str:
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 250},
        }
    ).encode("utf-8")
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent",
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    if not text:
        raise ValueError("Gemini returned no text")
    return text
