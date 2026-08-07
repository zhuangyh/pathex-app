"""
LLM-generated reasoning for already-extracted field values.

One call per case (not per field): given the report text and the fields
already extracted for that case, asks the model to explain, field by
field, what in the report supports each value -- explicitly distinguishing
a direct quote/synonym from an inference, unit conversion, or other
interpretation, and flagging a value it thinks the report doesn't actually
support. It's also asked for a short verbatim quote per field, which the
UI uses to highlight the supporting span in the original report.

This is the model explaining a decision in a fresh call, not a proof: it
can still rationalize a plausible-sounding explanation that doesn't
perfectly reflect how the value was originally produced. Treat it as a
starting point for review, not a guarantee. Every field name in the
returned explanations is one that was actually asked about -- nothing the
model invents for a field that wasn't in the request is passed through.
The quote is checked against the actual report text before use (see
find_quote_span) since the model can paraphrase instead of copying
verbatim, or invent a quote for an UNSUPPORTED field.
"""
from __future__ import annotations

import json
import re

_MIN_QUOTE_LEN = 3  # shorter quotes highlight too many false spans to be useful


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
        "below. For EACH field listed, explain what in the report supports that value.\n\n"
        "Be honest and specific:\n"
        "- If the value is a direct quote or a clear synonym of report text, say so and name "
        "the supporting phrase.\n"
        "- If it required inference, unit conversion, or interpretation (e.g. converting a "
        "stated size in cm to mm, inferring a stage from descriptive findings), say so "
        "explicitly and explain the inference.\n"
        "- If you don't think the report actually supports this value, start the explanation "
        'with "UNSUPPORTED:" and explain why, and use an empty string for the quote.\n\n'
        f"Fields and their extracted values:\n{lines}\n\n"
        "Return ONLY a JSON object of the form:\n"
        '{"<field name>": {"explanation": "<one-sentence explanation>", '
        '"quote": "<short verbatim span copied exactly from the report that most directly '
        'supports the value, a few words, not a full sentence>"}, ...}\n'
        "One entry for every field listed above, exact field names, no other text."
    )


def find_quote_span(report_text: str, quote: str) -> tuple[int, int] | None:
    """
    Locate `quote` inside `report_text`, tolerating whitespace differences
    (the model may collapse/normalize whitespace even when otherwise
    copying verbatim). Returns (start, end) offsets into report_text, or
    None if no reasonably exact match is found -- the model can also
    paraphrase or invent a quote for a field it declared UNSUPPORTED, and
    a stale/wrong span would highlight the wrong text, so a miss here
    should skip highlighting entirely rather than fall back to a fuzzy guess.
    """
    quote = (quote or "").strip()
    if len(quote) < _MIN_QUOTE_LEN:
        return None

    idx = report_text.find(quote)
    if idx != -1:
        return (idx, idx + len(quote))

    # Whitespace-normalized fallback: match on collapsed whitespace, then
    # map back to offsets in the original (unnormalized) report_text.
    norm_pattern = re.escape(quote)
    norm_pattern = re.sub(r"(?:\\ )+", r"\\s+", norm_pattern)
    m = re.search(norm_pattern, report_text, re.IGNORECASE)
    if m:
        return (m.start(), m.end())
    return None


def explain_case(report_text: str, field_values: dict, call_api_fn,
                  url: str, key: str, model: str, max_tokens: int = 2000) -> dict:
    """
    Returns {"explanations": {field: {"text": str, "quote": str|None,
    "span": (start, end)|None}}, "error": str|None}.

    `span` is only set when `quote` was actually located in report_text
    (see find_quote_span) -- the UI should only highlight when it's present.
    """
    if not field_values:
        return {"explanations": {}, "error": None}
    system_prompt = build_system_prompt(field_values)
    res = call_api_fn(report_text, system_prompt, url, key, model,
                       max_tokens=max_tokens, json_mode=True, temperature=0.0)
    if res.get("error"):
        return {"explanations": {}, "error": res["error"]}
    raw = _extract_object(res.get("message", ""))

    explanations = {}
    for f in field_values:
        entry = raw.get(f)
        if not entry:
            continue
        if isinstance(entry, dict):
            text  = str(entry.get("explanation", "")).strip()
            quote = str(entry.get("quote", "")).strip()
        else:
            # Tolerate a model that ignores the schema and returns a plain
            # string, as the previous prompt version asked for.
            text, quote = str(entry).strip(), ""
        if not text:
            continue
        span = find_quote_span(report_text, quote) if quote else None
        explanations[f] = {"text": text, "quote": quote or None, "span": span}

    return {"explanations": explanations, "error": None}
