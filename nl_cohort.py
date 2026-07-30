"""
Natural-language cohort-rule translator.

Translates a free-text cohort description (e.g. "liver cases with vascular
invasion and R1 margins") into structured inclusion/exclusion rules using
the exact field names, option vocabularies, and operators the Cohort tab
can actually execute (see OPERATORS / eval_condition in app.py) -- never
invents a field or operator outside that vocabulary.

Makes one LLM call to do the natural-language understanding (reusing the
app's own call_api, passed in rather than imported, to avoid a circular
import). Every rule the LLM proposes is re-validated against the real
field names and operator list before being returned; anything invented or
mismatched is silently dropped rather than passed through. This module
only *proposes* rules -- it never touches session state or applies
anything. The caller must show the proposal to the user for review before
ever writing it into ss.ie_rules.
"""
from __future__ import annotations

import json
import re


def _extract_rules(message: str) -> list:
    """Pull a JSON array out of the LLM's response, tolerating a {"rules": [...]}
    wrapper, a fenced code block, or trailing prose -- mirrors the same
    tiered-fallback approach app.py's own parse_json() uses."""
    text = (message or "").strip()

    def _as_list(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("rules"), list):
            return obj["rules"]
        return None

    try:
        got = _as_list(json.loads(text))
        if got is not None:
            return got
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if m:
        try:
            got = _as_list(json.loads(m.group(1)))
            if got is not None:
                return got
        except Exception:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        got = _as_list(json.loads(text[start:i + 1]))
                        if got is not None:
                            return got
                    except Exception:
                        pass
                    break
    return []


def _is_numeric_field(field: dict) -> bool:
    """True only for fields whose allowed-values vocabulary is a plain number
    (the app's own convention: options start with "Integer" or "Number").
    Everything else -- staging/grade codes, yes/no, free text -- is a fixed
    vocabulary that eval_condition can only compare with ==/!=/contains/in,
    never with >,>=,<,<=  (it parses the value as a float and any code like
    "pT3" or "GG3" simply fails to parse, silently matching zero rows)."""
    opts = str(field.get("options", ""))
    return opts.startswith("Integer") or opts.startswith("Number")


def build_system_prompt(fields: list[dict], operators: list[str]) -> str:
    field_lines = []
    for f in fields:
        tag = "numeric" if _is_numeric_field(f) else "fixed vocabulary, NOT numeric"
        field_lines.append(f'- "{f["name"]}" ({tag}) — allowed values: {f.get("options", "")}')
    fields_block = "\n".join(field_lines)
    ops_block = ", ".join(f'"{o}"' for o in operators)
    return (
        "You translate a researcher's plain-English cohort description into structured "
        "filter rules for a pathology-cohort tool.\n\n"
        "Only use these exact field names (verbatim, case-sensitive) — never invent a "
        "field that isn't listed. Each is tagged (numeric) or (fixed vocabulary, NOT "
        "numeric):\n"
        f"{fields_block}\n\n"
        f"Only use these exact operator strings: {ops_block}\n\n"
        'Return ONLY a JSON object: {"rules": [ {"field": "...", "operator": "...", '
        '"value": "...", "action": "..."}, ... ]}. No prose, no markdown fences.\n\n'
        "Guidance:\n"
        '- A criterion the cohort MUST satisfy (e.g. "with X", "positive for X") -> '
        '"action": "Include only if".\n'
        '- A criterion that disqualifies a case (e.g. "exclude X", "without X") -> '
        '"action": "Exclude if".\n'
        '- "is missing" and "is present" need no value -- use "".\n'
        "- If a phrase doesn't correspond to any listed field (e.g. it names the cancer "
        "type, which the dataset is already restricted to), omit it — do not invent a "
        "field or guess.\n"
        "- Translate the user's wording into the field's own allowed-values vocabulary "
        "above (e.g. \"R1 margins\" -> whichever exact option for that field means "
        "positive/involved margins).\n"
        "- For a numeric threshold (\"at least 3\", \"over 5 cm\") on a field tagged "
        "(numeric), use >, >=, <, or <= with a plain numeric value (no units, no %).\n"
        "- NEVER use >, >=, <, or <= on a field tagged (fixed vocabulary, NOT numeric) "
        "-- staging/grade codes like pT3, GG3, or N1 cannot be compared as numbers. "
        "Instead, for a threshold phrase against one of these fields (e.g. \"T-stage 3 "
        "or higher\", \"grade group 3 or above\"), expand it into every one of that "
        "field's own allowed values that satisfies the threshold, using clinical "
        "staging order (T1a/T1b/T1c < T2 < T3 < T4a/T4b/T4c/T4d, N0 < N1 < N2 < N3, "
        "GG1 < GG2 < ... < GG5, Grade 1 < Grade 2 < Grade 3), and return them as "
        'operator "in (comma list)" with a comma-separated value, e.g. "T-stage 3 or '
        'higher" over allowed values pT1a/pT1b/pT1c/pT2/pT3/pT4a/pT4b/pT4c/pT4d -> '
        '"operator": "in (comma list)", "value": "pT3,pT4a,pT4b,pT4c,pT4d".\n'
        "- Recognize well-known compound clinical shorthand and expand it into the "
        "individual field-level criteria it implies, using only the fields actually "
        'listed above (e.g. "triple-negative breast cancer" implies ER negative, PR '
        "negative, and HER2 negative -- emit one rule per field that exists in this "
        "field list; skip any implied criterion whose field isn't listed). Don't "
        "return an empty rule list just because the description uses a shorthand term "
        "instead of naming fields directly."
    )


def translate_query(query: str, fields: list[dict], operators: list[str],
                     call_api_fn, url: str, key: str, model: str,
                     max_tokens: int = 2000) -> dict:
    """
    Returns {"rules": [{"Field","Operator","Value","Action"}, ...],
             "n_dropped": int, "error": str|None, "raw_message": str}.
    n_dropped counts LLM-proposed rules that referenced an unknown field or
    operator, or used a numeric comparison (>,>=,<,<=) on a non-numeric
    field, and were discarded rather than passed through.
    """
    system_prompt = build_system_prompt(fields, operators)
    res = call_api_fn(query, system_prompt, url, key, model,
                       max_tokens=max_tokens, json_mode=True, temperature=0.0)
    if res.get("error"):
        return {"rules": [], "n_dropped": 0, "error": res["error"], "raw_message": ""}

    message = res.get("message", "")
    raw_rules = _extract_rules(message)

    field_names = {f["name"] for f in fields}
    numeric_fields = {f["name"] for f in fields if _is_numeric_field(f)}
    op_set = set(operators)
    ordering_ops = {">", ">=", "<", "<="}
    valid_rules, n_dropped = [], 0
    for r in raw_rules:
        if not isinstance(r, dict):
            n_dropped += 1
            continue
        field = r.get("field")
        op = r.get("operator")
        if field not in field_names or op not in op_set:
            n_dropped += 1
            continue
        if op in ordering_ops and field not in numeric_fields:
            # Despite the prompt's guidance, the model used a numeric
            # comparison on a staging/grade code (e.g. ">= pT3") -- eval_condition
            # can never match this (it parses the value as a float), so it
            # would silently pass validation yet always match zero rows.
            # Drop it visibly instead of shipping a rule that can never fire.
            n_dropped += 1
            continue
        action = r.get("action")
        if action not in ("Include only if", "Exclude if"):
            action = "Include only if"
        value = r.get("value", "")
        valid_rules.append({
            "Field": field, "Operator": op,
            "Value": "" if value is None else str(value),
            "Action": action,
        })
    return {"rules": valid_rules, "n_dropped": n_dropped, "error": None, "raw_message": message}
