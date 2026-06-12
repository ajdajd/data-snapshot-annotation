import base64
from dotenv import load_dotenv
import json
from pathlib import Path
import os
import re
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# USD per 1M tokens
PRICING = {
    "gpt-5.5": {
        "input_per_1M": 5.00,
        "output_per_1M": 30.00,
    },
    "gpt-5.4": {
        "input_per_1M": 2.50,
        "output_per_1M": 15.00,
    },
    "gpt-5.4-mini": {
        "input_per_1M": 0.75,
        "output_per_1M": 4.50,
    },
}


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def compute_cost(model: str, usage) -> dict:
    """Compute the USD cost of an API call from a usage object.

    Parameters
    ----------
    model : str
        Model name (must exist in ``PRICING``).
    usage : ResponseUsage
        The ``response.usage`` object returned by the OpenAI Responses API.

    Returns
    -------
    dict
        Token counts and cost breakdown in USD.
    """
    pricing = PRICING[model]

    input_cost = (usage.input_tokens / 1e6) * pricing["input_per_1M"]
    output_cost = (usage.output_tokens / 1e6) * pricing["output_per_1M"]

    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def _encode_image_to_data_url(image_path: str) -> str:
    """Read a local image file and return a base64 data URL.

    Parameters
    ----------
    image_path : str
        Path to the image file (PNG, JPEG, etc.).

    Returns
    -------
    str
        A ``data:image/<ext>;base64,...`` URL string.
    """
    path = Path(image_path)
    suffix = path.suffix.lstrip(".").lower()
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime = mime_map.get(suffix, f"image/{suffix}")

    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping a JSON block.

    Parameters
    ----------
    text : str
        Raw LLM output text.

    Returns
    -------
    str
        Text with leading/trailing code fences removed.
    """
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def analyze_snapshot(
    system_prompt: str,
    user_prompt: str,
    image_path: str,
    model: str = "gpt-5.4-mini",
    max_output_tokens: int = 3000,
) -> dict:
    """Send a snapshot image to the OpenAI Responses API.

    Parameters
    ----------
    system_prompt : str
        System-level instructions for the model.
    user_prompt : str
        Rendered user prompt with placeholders filled.
    image_path : str
        Path to the snapshot image file.
    model : str
        OpenAI model name.
    max_output_tokens : int
        Maximum tokens for the model response.

    Returns
    -------
    Response
    """
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": _encode_image_to_data_url(image_path),
                    },
                ],
            },
        ],
    )

    return response


def process_response(response, model):
    """Parse Response object and add usage details.

    Parameters
    ----------
    response : Response
        Response object to parse
    model : str
        OpenAI model name.

    Returns
    -------
    dict
        Always contains ``parsed_output``, ``raw_output``, ``usage``,
        ``cost``, and ``error``. On parse failure, ``parsed_output`` is
        ``None`` and ``error`` describes the issue.
    """
    try:
        raw_output = response.output_text.strip()
        usage = response.usage
        cost = compute_cost(model, usage)

    except Exception as e:
        return {
            "parsed_output": None,
            "raw_output": raw_output,
            "usage": usage,
            "cost": cost,
            "error": f"{e}",
        }

    return {
        "parsed_output": None,
        "raw_output": raw_output,
        "usage": usage,
        "cost": cost,
        "error": None,
    }
