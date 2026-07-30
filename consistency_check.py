"""
Local, deterministic cross-field consistency checker.

Runs entirely on already-extracted structured field values -- no LLM calls,
no report text, no network. It flags internal contradictions in a case's
extracted fields, e.g. "N0" (no nodal metastasis, by definition) paired
with a nonzero positive-node count, or more positive nodes than nodes
examined (a mathematical impossibility).

Every check here is true by construction, not a matter of clinical
judgment -- deliberately narrow in scope. A tempting but excluded check was
"N-stage indicates nodal involvement but positive-node count is 0": AJCC
colorectal N1c specifically means a tumor deposit *without* conventional
lymph node metastasis, so that direction has real false-positive risk and
was left out rather than shipped with unverified accuracy.

This is a QC/review aid, not a correction mechanism: it never modifies
extracted values, and its output must stay visually separate from the
extraction results (a flag is a "look at this," not a new data field).
"""
from __future__ import annotations

import re

import pandas as pd

_TYPE_TOKENS = {"integer", "number", "string"}


def _split_options(options: str) -> list[str]:
    return [o.strip() for o in (options or "").split("/") if o.strip()]


def _is_numeric_like(value: str) -> bool:
    s = value.strip().rstrip("%")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", s):  # e.g. "0/12"
        return True
    return False


def check_out_of_vocabulary(fields: list[dict], row: dict) -> list[dict]:
    """
    Flag any field whose extracted value matches neither "Not mentioned"
    nor one of the options the field itself defines. Skips free-text
    ("String") fields, which have no fixed vocabulary to check against.
    """
    flags = []
    for f in fields:
        name = f["name"]
        if name not in row:
            continue
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if value.lower() == "not mentioned":
            continue

        opts = _split_options(f.get("options", ""))
        opt_tokens = {o.lower() for o in opts}

        if "integer" in opt_tokens or "number" in opt_tokens:
            if not _is_numeric_like(value):
                flags.append({
                    "check": "Out-of-vocabulary",
                    "field": name,
                    "message": f"Expected a number (or \"Not mentioned\"), got {value!r}.",
                })
            continue
        if "string" in opt_tokens or not opts:
            continue  # free text -- nothing to validate against

        if value.lower() not in opt_tokens:
            flags.append({
                "check": "Out-of-vocabulary",
                "field": name,
                "message": f"{value!r} is not one of the defined options for this field "
                           f"({f.get('options', '')}).",
            })
    return flags


def _find_field(fields: list[dict], *, must_contain: list[str]) -> str | None:
    """Case-insensitive substring match on field name; returns the first hit."""
    for f in fields:
        n = f["name"].lower()
        if all(term in n for term in must_contain):
            return f["name"]
    return None


def _to_int(value) -> int | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    return None


def check_node_math(fields: list[dict], row: dict) -> list[dict]:
    """Flag Lymph Nodes Positive > Lymph Nodes Examined -- impossible by definition."""
    examined_field = _find_field(fields, must_contain=["lymph", "examined"])
    positive_field = (_find_field(fields, must_contain=["lymph", "positive"])
                       or _find_field(fields, must_contain=["lymph", "metastasis"]))
    if not examined_field or not positive_field:
        return []

    examined = _to_int(row.get(examined_field))
    positive = _to_int(row.get(positive_field))
    if examined is None or positive is None:
        return []

    if positive > examined:
        return [{
            "check": "Node-count math",
            "field": f"{positive_field} / {examined_field}",
            "message": f"{positive_field}={positive} exceeds {examined_field}={examined} "
                       "-- more positive nodes than nodes examined is not possible.",
        }]
    return []


def check_n0_contradiction(fields: list[dict], row: dict) -> list[dict]:
    """Flag N-Stage == N0 (zero nodal metastasis, by definition) with a nonzero positive-node count."""
    n_field = _find_field(fields, must_contain=["n-stage"]) or _find_field(fields, must_contain=["n stage"])
    positive_field = (_find_field(fields, must_contain=["lymph", "positive"])
                       or _find_field(fields, must_contain=["lymph", "metastasis"]))
    if not n_field or not positive_field:
        return []

    n_raw = row.get(n_field)
    if not isinstance(n_raw, str):
        return []
    n_norm = re.sub(r"^p", "", n_raw.strip(), flags=re.IGNORECASE).upper()
    if n_norm != "N0":
        return []

    positive = _to_int(row.get(positive_field))
    if positive is None or positive == 0:
        return []

    return [{
        "check": "N0 contradiction",
        "field": f"{n_field} / {positive_field}",
        "message": f"{n_field}=N0 (no nodal metastasis, by definition) but "
                   f"{positive_field}={positive}.",
    }]


_CHECKS = [check_out_of_vocabulary, check_node_math, check_n0_contradiction]


def check_row(fields: list[dict], row: dict) -> list[dict]:
    flags = []
    for check in _CHECKS:
        flags.extend(check(fields, row))
    return flags


def check_dataframe(fields: list[dict], results_df: pd.DataFrame) -> pd.DataFrame:
    """One row per flag across all cases: Case_ID, Check, Field, Message."""
    records = []
    for _, row in results_df.iterrows():
        row_dict = row.to_dict()
        for flag in check_row(fields, row_dict):
            records.append({
                "Case_ID": row_dict.get("Case_ID"),
                "Check": flag["check"],
                "Field": flag["field"],
                "Message": flag["message"],
            })
    return pd.DataFrame(records, columns=["Case_ID", "Check", "Field", "Message"])
