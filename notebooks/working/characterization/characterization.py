from dotenv import load_dotenv
import json
from pathlib import Path
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# USD per 1M tokens
PRICING = {
    "gpt-4o-mini": {
        "input_per_1M": 0.15,
        "output_per_1M": 0.60,
    }
}


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_user_prompt(template: str, metadata: dict) -> str:
    text = template
    for key, value in metadata.items():
        placeholder = f"{{{{{key}}}}}"
        text = text.replace(placeholder, str(value) if value is not None else "unknown")
    return text


def compute_cost(model: str, usage: dict) -> dict:
    """
    usage example:
    {
        'input_tokens': 987,
        'output_tokens': 142,
        'total_tokens': 1129
    }
    """
    pricing = PRICING[model]

    input_cost = (usage["input_tokens"] / 1e6) * pricing["input_per_1k"]
    output_cost = (usage["output_tokens"] / 1e6) * pricing["output_per_1k"]

    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def analyze_snapshot(
    system_prompt: str,
    user_prompt: str,
    image_path: str,
    model: str = "gpt-4o-mini",
    max_output_tokens: int = 300,
) -> dict:
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"file://{Path(image_path).absolute()}",
                    },
                ],
            },
        ],
        max_output_tokens=max_output_tokens,
    )

    output_text = response.output_text.strip()

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON returned:\n{output_text}")

    usage = response.usage
    cost = compute_cost(model, usage)

    return {
        "parsed_output": parsed,
        "usage": usage,
        "cost": cost,
    }
