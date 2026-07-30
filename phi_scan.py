"""
Local, regex-based PHI pre-pass.

Runs entirely on-device before report text is sent to any LLM provider --
no network calls, no API cost, no new dependency. This is a heuristic
advisory scan, not a HIPAA Safe Harbor de-identification guarantee: it
looks for common identifier *patterns* (SSNs, phone numbers, emails,
labeled MRN/DOB/name fields, street addresses) but cannot catch every way
a name or date could appear in free text. A clean scan means "no obvious
red flags," not certified de-identification -- the user is still
responsible for uploading only de-identified or synthetic reports.
"""
from __future__ import annotations

import re

import pandas as pd

# Each entry: (PHI type label, compiled regex). Where possible the pattern
# requires a labeling keyword (e.g. "MRN:", "DOB:") rather than a bare
# digit/date pattern, since pathology reports legitimately contain plenty
# of unlabeled numbers and dates (specimen sizes, receive/report dates,
# tumor measurements) that are not identifiers and would otherwise flood
# the scan with false positives.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # \b doesn't fire directly before a literal "(" (both the preceding
    # space and "(" are non-word characters, so there's no word/non-word
    # transition for \b to anchor on) -- \b is scoped to just the
    # digits-only alternative instead.
    ("Phone number", re.compile(r"(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b")),
    ("Email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("Medical record / account number",
     re.compile(r"\b(?:MRN|Medical\s*Record\s*(?:No\.?|Number|#)|Account\s*(?:No\.?|#)|"
                r"Accession\s*(?:No\.?|#))\s*[:#]?\s*[\w-]{4,}", re.IGNORECASE)),
    # Label matching is case-insensitive ((?i:...) scopes the flag to just
    # that group); the name value itself is required to be capitalized and
    # excludes common placeholder fillers ("Not mentioned", "N/A", etc.)
    # that would otherwise false-positive as if they were a real name.
    ("Patient name field",
     re.compile(r"\b(?i:Patient\s*Name|Pt\.?\s*Name|Name\s*of\s*Patient)\s*[:#]\s*"
                r"(?!Not\s+mentioned\b|N/?A\b|Unknown\b|Anonymous\b|Redacted\b)"
                r"[A-Z][\w'.-]{1,30}(?:\s+[A-Z][\w'.-]{1,30}){0,3}")),
    ("Date of birth",
     re.compile(r"\b(?:DOB|Date\s*of\s*Birth|Birth\s*Date)\s*[:#]?\s*"
                r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", re.IGNORECASE)),
    # Case-sensitive on purpose: with IGNORECASE this collides with common
    # pathology abbreviations that happen to share letters with street
    # suffixes -- "LN" (lymph node) matching "Ln" (Lane) was the motivating
    # false positive found while testing against the app's own sample data.
    # "Way" and "Court" were dropped entirely -- too common in plain prose
    # to serve as a reliable street-suffix signal even capitalized.
    ("Street address",
     re.compile(r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){1,4}"
                r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?)\b")),
    ("ZIP+4 code", re.compile(r"\b\d{5}-\d{4}\b")),
]


def scan_text(text: str) -> list[dict]:
    """Return one {type, match, start, end} dict per PHI-like hit in text."""
    if not text:
        return []
    hits = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            hits.append({"type": label, "match": m.group(0), "start": m.start(), "end": m.end()})
    return hits


def scan_dataframe(df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    """
    Scan every row's report text and return one row per hit, for display
    in the UI: Case ID, PHI type, and the matched snippet.
    """
    records = []
    for _, row in df.iterrows():
        text = row.get(text_col)
        if not isinstance(text, str):
            continue
        for hit in scan_text(text):
            records.append({
                "Case ID": row.get(id_col),
                "PHI type": hit["type"],
                "Snippet": hit["match"].strip(),
            })
    return pd.DataFrame(records, columns=["Case ID", "PHI type", "Snippet"])
