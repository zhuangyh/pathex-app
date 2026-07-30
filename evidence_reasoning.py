"""
LLM-generated reasoning for already-extracted field values.

One call per case (not per field): given the report text and the fields
already extracted for that case, asks the model to explain, field by
field, what in the report supports each value -- explicitly distinguishing
a direct quote/synonym from an inference, unit conversion, or other
interpretation, and flagging a value it thinks the report doesn't actually
support.

This is the model explaining a decision in a fresh call, not a proof: it
can still rationalize a plausible-sounding explanation that doesn't
perfectly reflect how the value was originally produced. Treat it as a
starting point for review, not a guarantee. Every field name in the
returned explanations is one that was actually asked about -- nothing the
model invents for a field that wasn't in the request is passed through.
"""
from __future__ import annotations

import json
import re


def _extract_object(message: str) -> dict:
    text = (message or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
    return {}


def build_system_prompt(field_values: dict) -> str:
    lines = "\n".join(f'- "{f}": {v!r}' for f, v in field_values.items())
    return (
        "You previously extracted structured field values from the pathology report given "
        "below. For EACH field listed, write one short sentence explaining what in the report "
        "supports that value.\n\n"
        "Be honest and specific:\n"
        "- If the value is a direct quote or a clear synonym of report text, say so and name "
        "the supporting phrase.\n"
        "- If it required inference, unit conversion, or interpretation (e.g. converting a "
        "stated size in cm to mm, inferring a stage from descriptive findings), say so "
        "explicitly and explain the inference.\n"
        "- If you don't think the report actually supports this value, start the sentence with "
        '"UNSUPPORTED:" and explain why.\n\n'
        f"Fields and their extracted values:\n{lines}\n\n"
        'Return ONLY a JSON object: {"<field name>": "<one-sentence explanation>", ...} -- one '
        "entry for every field listed above, exact field names, no other text."
    )


def explain_case(report_text: str, field_values: dict, call_api_fn,
                  url: str, key: str, model: str, max_tokens: int = 2000) -> dict:
    """
    Returns {"explanations": {field: text}, "error": str|None}.
    """
    if not field_values:
        return {"explanations": {}, "error": None}
    system_prompt = build_system_prompt(field_values)
    res = call_api_fn(report_text, system_prompt, url, key, model,
                       max_tokens=max_tokens, json_mode=True, temperature=0.0)
    if res.get("error"):
        return {"explanations": {}, "error": res["error"]}
    raw = _extract_object(res.get("message", ""))
    explanations = {f: str(raw[f]) for f in field_values if raw.get(f)}
    return {"explanations": explanations, "error": None}
