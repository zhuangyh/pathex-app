"""
PathEx — Pathology Structured Extractor (Streamlit edition, multi-user)

Authors: Yonghua Zhuang, Vida Alami, William J. Feser, Dexiang Gao
License: MIT (see LICENSE)
Citation: Zhuang Y, Alami V, Feser WJ, Gao D. PathEx: A Web-Based Multi-Strategy Platform for
          Large Language Model-Driven Structured Extraction from Pathology Reports.
          (manuscript in preparation)
"""

from __future__ import annotations

import base64
import hashlib, html, json, os, re, time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from urllib.parse import quote
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import consistency_check
import evidence_reasoning
import nl_cohort
import phi_scan
from auth import (sidebar_auth_widget, require_login_for_action,
                  save_user_state, load_user_state,
                  save_user_csv, load_user_csv, user_dir,
                  save_user_secret, load_user_secret, clear_user_secret)

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
SHARED_KEY     = os.getenv("SHARED_API_KEY", "")
SHARED_MAX     = int(os.getenv("SHARED_KEY_MAX_CALLS", "50"))
SHARED_MODEL   = os.getenv("SHARED_KEY_MODEL", "gpt-4o-mini")
SHARED_URL     = os.getenv("SHARED_KEY_URL", "https://api.openai.com/v1/chat/completions")
SHARED_ENABLED = bool(SHARED_KEY)

# ── Provider catalogue ─────────────────────────────────────────────────────────
PROVIDERS = {
    "OpenAI": {
        "url":    "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-4o-mini", "gpt-4", "gpt-4.1", "gpt-5", "Custom…"],
        "key_help": "API key from platform.openai.com",
        "key_placeholder": "sk-…",
    },
    "Google Gemini": {
        "url":    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        # The dated preview/versioned names (gemini-2.5-flash-preview-05-20,
        # gemini-2.0-flash, gemini-1.5-*, etc.) have since been retired by
        # Google (404). The "-latest" aliases stay current as Google rotates
        # the underlying model.
        "models": ["gemini-flash-latest", "gemini-pro-latest", "Custom…"],
        "key_help": "API key from aistudio.google.com",
        "key_placeholder": "AIzaSy… (not sk-…)",
    },
    "OpenRouter (Claude / Gemini / Llama…)": {
        "url":    "https://openrouter.ai/api/v1/chat/completions",
        "models": [
            "openai/gpt-5",
            "anthropic/claude-opus-5",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-opus-4",
            "anthropic/claude-haiku-4-5",
            "google/gemini-2.5-pro-preview",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-large",
            "Custom…",
        ],
        "key_help": "API key from openrouter.ai/keys",
        "key_placeholder": "sk-or-v1-…",
    },
    "Azure OpenAI": {
        "url":    "https://<your-resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-02-01",
        "models": ["Custom…"],
        "key_help": "Bearer token from `az account get-access-token`",
        "key_placeholder": "eyJ0eXAi… (bearer token, not sk-…)",
    },
}

# Flat catalog of known models for the "Compare Models" checkbox list — one
# entry per (provider, model), excluding "Custom…" and Azure (its URL is
# per-deployment, so it doesn't fit a fixed catalog).
MODEL_CATALOG = [
    {"id": f"{prov}::{model}", "provider": prov, "model": model, "url": cfg["url"]}
    for prov, cfg in PROVIDERS.items()
    if prov != "Azure OpenAI"
    for model in cfg["models"]
    if model != "Custom…"
]

# ── Live model-list fetching ────────────────────────────────────────────────
# Keeps the dropdown current without a code change whenever a provider ships
# a new model — the hardcoded lists above remain as the offline fallback
# (and the only option for Azure, whose deployment names aren't a public
# catalog at all). Cached for 10 minutes per (provider, key) so repeated
# reruns/clicks don't hammer the provider's API.
_MODEL_ID_EXCLUDE = ("embedding", "whisper", "tts", "dall-e", "moderation",
                     "audio", "realtime", "transcribe", "image")

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_openai_models(api_key: str) -> list[str]:
    r = requests.get("https://api.openai.com/v1/models",
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    keep = [i for i in ids
            if any(p in i for p in ("gpt", "o1", "o3", "o4", "chatgpt"))
            and not any(x in i for x in _MODEL_ID_EXCLUDE)]
    return sorted(set(keep))

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_gemini_models(api_key: str) -> list[str]:
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                      params={"key": api_key}, timeout=10)
    r.raise_for_status()
    models = r.json().get("models", [])
    keep = [m["name"].removeprefix("models/") for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])]
    return sorted(set(keep))

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_openrouter_models(_api_key: str = "") -> list[str]:
    # OpenRouter's catalog endpoint is public — no key required — but the
    # (unused) _api_key param keeps this function's signature consistent
    # with the other two fetchers for the call site below.
    r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    return sorted(set(ids))

_MODEL_FETCHERS = {
    "OpenAI":          _fetch_openai_models,
    "Google Gemini":   _fetch_gemini_models,
    "OpenRouter (Claude / Gemini / Llama…)": _fetch_openrouter_models,
}

# Models known to reject the temperature parameter
_NO_TEMP_PREFIXES = (
    "o1", "o3", "o4",          # OpenAI reasoning series
    "gpt-5",                   # GPT-5 reasoning
    "gemini-2.5",              # Gemini 2.5 thinking models
)

PRESETS = {
    # ── Prostate (Grothey et al. 2025) ───────────────────────────────────────
    "Prostate Cancer — PRAD (Grothey et al. 2025)": {"fields": [
        {"name": "WHO (ISUP) Grade Group",                  "options": "GG1/GG2/GG3/GG4/GG5/Not mentioned",               "gt_col": "WHO_GG"},
        {"name": "T-Stage (TNM)",                           "options": "pT0/pT2a/pT2b/pT2c/pT3a/pT3b/pT4/Not mentioned",  "gt_col": "pT"},
        {"name": "N-Stage (TNM)",                           "options": "pNx/pN0/pN1/Not mentioned",                         "gt_col": "pN"},
        {"name": "Number of Lymph Nodes examined",          "options": "Integer/Not mentioned",                             "gt_col": "LYM_POS", "gt_norm": "lym_total"},
        {"name": "Lymph Nodes with Metastasis",             "options": "Integer/Not mentioned",                             "gt_col": "LYM_POS", "gt_norm": "lym_pos"},
        {"name": "Resection Margins",                       "options": "R0/R1/Rx/Not mentioned",                           "gt_col": "R"},
        {"name": "Histologic Subtype",                      "options": "acinar/ductal/mixed/Not mentioned",                 "gt_col": "SUBTYPE"},
        {"name": "Primary Gleason Pattern",                 "options": "3/4/5/Not mentioned",                              "gt_col": "GP_1",     "gt_norm": "integer"},
        {"name": "Secondary Gleason Pattern",               "options": "3/4/5/Not mentioned",                              "gt_col": "GP_2",     "gt_norm": "integer"},
        {"name": "Tertiary Gleason Pattern",                "options": "3/4/5/Not mentioned",                              "gt_col": "GP_3",     "gt_norm": "gp3"},
        {"name": "Percentage of Secondary Gleason Pattern", "options": "Integer/Not mentioned",                            "gt_col": "GP_2_PERC","gt_norm": "pct"},
    ]},

    # ── Thyroid (Lee et al. 2025 — 12-question MQA framework) ────────────────
    "Thyroid Cancer (Lee et al. 2025)": {"fields": [
        {"name": "Primary Tumor Size (cm)",          "options": "Number/Not mentioned"},
        {"name": "Extrathyroidal Extension",          "options": "yes/no/Not mentioned"},
        {"name": "Primary Tumor Location",            "options": "right lobe/left lobe/isthmus/bilateral/Not mentioned"},
        {"name": "Cervical Lymph Nodes Present",      "options": "yes/no/Not mentioned"},
        {"name": "Lymph Nodes Positive for Malignancy","options": "Integer/Not mentioned"},
        {"name": "TNM Stage",                         "options": "String/Not mentioned"},
        {"name": "Vascular Invasion",                 "options": "present/absent/Not mentioned"},
        {"name": "Lymphatic Invasion",                "options": "present/absent/Not mentioned"},
        {"name": "Histology",                         "options": "papillary/follicular/medullary/anaplastic/Hurthle cell/poorly differentiated/Not mentioned"},
        {"name": "Aggressive Histologic Variant",     "options": "tall cell/hobnail/columnar cell/diffuse sclerosing/solid/none/Not mentioned"},
        {"name": "Second Thyroid Cancer Present",     "options": "yes/no/Not mentioned"},
        {"name": "Histology of Second Cancer",        "options": "String/Not mentioned"},
    ]},

    # ── Breast Cancer (CAP / AJCC 8th ed.) ───────────────────────────────────
    "Breast Cancer (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",              "options": "invasive ductal/invasive lobular/mixed ductal-lobular/mucinous/tubular/cribriform/micropapillary/metaplastic/other/Not mentioned"},
        {"name": "Nottingham Grade",             "options": "Grade 1/Grade 2/Grade 3/Not mentioned"},
        {"name": "Tumor Size (mm)",              "options": "Number/Not mentioned"},
        {"name": "T-Stage (TNM)",                "options": "pTis/pT1a/pT1b/pT1c/pT2/pT3/pT4a/pT4b/pT4c/pT4d/Not mentioned"},
        {"name": "N-Stage (TNM)",                "options": "pN0/pN1mi/pN1a/pN1b/pN1c/pN2a/pN2b/pN3a/pN3b/pN3c/Not mentioned"},
        {"name": "Lymph Nodes Examined",         "options": "Integer/Not mentioned"},
        {"name": "Lymph Nodes Positive",         "options": "Integer/Not mentioned"},
        {"name": "Surgical Margins",             "options": "negative/positive/close/not assessed/Not mentioned"},
        {"name": "Estrogen Receptor (ER)",       "options": "positive/negative/Not mentioned"},
        {"name": "Progesterone Receptor (PR)",   "options": "positive/negative/Not mentioned"},
        {"name": "HER2 Status",                  "options": "positive/negative/equivocal/Not mentioned"},
        {"name": "Ki-67 (%)",                    "options": "Number/Not mentioned"},
        {"name": "Lymphovascular Invasion",      "options": "present/absent/Not mentioned"},
        {"name": "DCIS Component",               "options": "present/absent/Not mentioned"},
    ]},

    # ── Breast Cancer — TCGA-242 benchmark variant ───────────────────────────
    # Same as "Breast Cancer (CAP/AJCC)" minus Ki-67, which the TCGA-242 gold
    # schema never records (0 evaluable cases in tcga242_report.md §6.1).
    "Breast Cancer — TCGA-242 (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",              "options": "invasive ductal/invasive lobular/mixed ductal-lobular/mucinous/tubular/cribriform/micropapillary/metaplastic/other/Not mentioned", "gt_col": "Histologic Type"},
        {"name": "Nottingham Grade",             "options": "Grade 1/Grade 2/Grade 3/Not mentioned",                       "gt_col": "Nottingham Grade"},
        {"name": "Tumor Size (mm)",              "options": "Number/Not mentioned",                                        "gt_col": "Tumor Size (mm)"},
        {"name": "T-Stage (TNM)",                "options": "pTis/pT1a/pT1b/pT1c/pT2/pT3/pT4a/pT4b/pT4c/pT4d/Not mentioned", "gt_col": "T-Stage (TNM)"},
        {"name": "N-Stage (TNM)",                "options": "pN0/pN1mi/pN1a/pN1b/pN1c/pN2a/pN2b/pN3a/pN3b/pN3c/Not mentioned", "gt_col": "N-Stage (TNM)"},
        {"name": "Lymph Nodes Examined",         "options": "Integer/Not mentioned",                                       "gt_col": "Lymph Nodes Examined"},
        {"name": "Lymph Nodes Positive",         "options": "Integer/Not mentioned",                                       "gt_col": "Lymph Nodes Positive"},
        {"name": "Surgical Margins",             "options": "negative/positive/close/not assessed/Not mentioned",          "gt_col": "Surgical Margins"},
        {"name": "Estrogen Receptor (ER)",       "options": "positive/negative/Not mentioned",                             "gt_col": "Estrogen Receptor (ER)"},
        {"name": "Progesterone Receptor (PR)",   "options": "positive/negative/Not mentioned",                             "gt_col": "Progesterone Receptor (PR)"},
        {"name": "HER2 Status",                  "options": "positive/negative/equivocal/Not mentioned",                  "gt_col": "HER2 Status"},
        {"name": "Lymphovascular Invasion",      "options": "present/absent/Not mentioned",                                "gt_col": "Lymphovascular Invasion"},
        {"name": "DCIS Component",               "options": "present/absent/Not mentioned",                               "gt_col": "DCIS Component"},
    ]},

    # ── Colorectal Cancer (CAP / AJCC 8th ed.) ───────────────────────────────
    "Colorectal Cancer (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",              "options": "adenocarcinoma/mucinous adenocarcinoma/signet ring cell/serrated adenocarcinoma/undifferentiated/other/Not mentioned"},
        {"name": "Histologic Grade",             "options": "well differentiated/moderately differentiated/poorly differentiated/undifferentiated/Not mentioned"},
        {"name": "T-Stage (TNM)",                "options": "pT1/pT2/pT3/pT4a/pT4b/Not mentioned"},
        {"name": "N-Stage (TNM)",                "options": "pN0/pN1a/pN1b/pN1c/pN2a/pN2b/Not mentioned"},
        {"name": "M-Stage (TNM)",                "options": "pM0/pM1a/pM1b/pM1c/Not mentioned"},
        {"name": "Lymph Nodes Examined",         "options": "Integer/Not mentioned"},
        {"name": "Lymph Nodes Positive",         "options": "Integer/Not mentioned"},
        {"name": "Resection Margins",            "options": "R0/R1/R2/Not mentioned"},
        {"name": "Circumferential Resection Margin", "options": "clear/involved/Not mentioned"},
        {"name": "Lymphovascular Invasion",      "options": "present/absent/Not mentioned"},
        {"name": "Perineural Invasion",          "options": "present/absent/Not mentioned"},
        {"name": "Tumor Deposits",               "options": "present/absent/Not mentioned"},
        {"name": "Microsatellite Status",        "options": "MSS/MSI-L/MSI-H/Not mentioned"},
        {"name": "Tumour Budding",               "options": "low/intermediate/high/Not mentioned"},
    ]},

    # ── Colorectal Cancer — TCGA-242 benchmark variant ───────────────────────
    # Same as "Colorectal Cancer (CAP/AJCC)" minus Tumor Deposits, Microsatellite
    # Status, and Tumour Budding — none are captured in the TCGA-242 gold schema
    # (0 evaluable cases each in tcga242_report.md §6.2) — and minus M-Stage,
    # removed from scoring because misses dominated its errors (≥50%) — gold
    # computes it from criteria the report text never states as an extractable
    # token (tcga242_report_summary.md, "Some T/N/M-Stage fields were removed
    # entirely"; full detail in tcga242_report.md §9.11).
    "Colorectal Cancer — TCGA-242 (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",              "options": "adenocarcinoma/mucinous adenocarcinoma/signet ring cell/serrated adenocarcinoma/undifferentiated/other/Not mentioned", "gt_col": "Histologic Type"},
        {"name": "Histologic Grade",             "options": "well differentiated/moderately differentiated/poorly differentiated/undifferentiated/Not mentioned", "gt_col": "Histologic Grade"},
        {"name": "T-Stage (TNM)",                "options": "pT1/pT2/pT3/pT4a/pT4b/Not mentioned",                        "gt_col": "T-Stage (TNM)"},
        {"name": "N-Stage (TNM)",                "options": "pN0/pN1a/pN1b/pN1c/pN2a/pN2b/Not mentioned",                 "gt_col": "N-Stage (TNM)"},
        {"name": "Lymph Nodes Examined",         "options": "Integer/Not mentioned",                                     "gt_col": "Lymph Nodes Examined"},
        {"name": "Lymph Nodes Positive",         "options": "Integer/Not mentioned",                                     "gt_col": "Lymph Nodes Positive"},
        {"name": "Resection Margins",            "options": "R0/R1/R2/Not mentioned",                                    "gt_col": "Resection Margins"},
        {"name": "Circumferential Resection Margin", "options": "clear/involved/Not mentioned",                          "gt_col": "Circumferential Resection Margin"},
        {"name": "Lymphovascular Invasion",      "options": "present/absent/Not mentioned",                              "gt_col": "Lymphovascular Invasion"},
        {"name": "Perineural Invasion",          "options": "present/absent/Not mentioned",                              "gt_col": "Perineural Invasion"},
    ]},

    # ── Esophagus Cancer (CAP / AJCC 8th ed.) ────────────────────────────────
    "Esophagus Cancer (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",         "options": "squamous cell carcinoma/adenocarcinoma/other/Not mentioned"},
        {"name": "Histologic Grade",        "options": "well differentiated/moderately differentiated/poorly differentiated/Not mentioned"},
        {"name": "Primary Tumor Site",      "options": "upper third/middle third/lower third/gastroesophageal junction/Not mentioned"},
        {"name": "T-Stage (TNM)",           "options": "pT1a/pT1b/pT2/pT3/pT4a/pT4b/Not mentioned"},
        {"name": "N-Stage (TNM)",           "options": "pN0/pN1/pN2/pN3/Not mentioned"},
        {"name": "M-Stage (TNM)",           "options": "pM0/pM1/Not mentioned"},
        {"name": "Lymph Nodes Examined",    "options": "Integer/Not mentioned"},
        {"name": "Lymph Nodes Positive",    "options": "Integer/Not mentioned"},
        {"name": "Surgical Margins",        "options": "negative/positive/Not mentioned"},
        {"name": "Lymphovascular Invasion", "options": "present/absent/Not mentioned"},
        {"name": "Perineural Invasion",     "options": "present/absent/Not mentioned"},
    ]},

    # ── Esophagus Cancer — TCGA-242 benchmark variant ────────────────────────
    # Same as "Esophagus Cancer (CAP/AJCC)" minus T-Stage and M-Stage, both
    # removed from scoring because misses dominated their errors (≥50%) — gold
    # computes these from criteria the report text never states as an
    # extractable token (tcga242_report_summary.md, "Some T/N/M-Stage fields
    # were removed entirely"; full detail in tcga242_report.md §9.11).
    "Esophagus Cancer — TCGA-242 (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",         "options": "squamous cell carcinoma/adenocarcinoma/other/Not mentioned",  "gt_col": "Histologic Type"},
        {"name": "Histologic Grade",        "options": "well differentiated/moderately differentiated/poorly differentiated/Not mentioned", "gt_col": "Histologic Grade"},
        {"name": "Primary Tumor Site",      "options": "upper third/middle third/lower third/gastroesophageal junction/Not mentioned", "gt_col": "Primary Tumor Site"},
        {"name": "N-Stage (TNM)",           "options": "pN0/pN1/pN2/pN3/Not mentioned",                                "gt_col": "N-Stage (TNM)"},
        {"name": "Lymph Nodes Examined",    "options": "Integer/Not mentioned",                                        "gt_col": "Lymph Nodes Examined"},
        {"name": "Lymph Nodes Positive",    "options": "Integer/Not mentioned",                                        "gt_col": "Lymph Nodes Positive"},
        {"name": "Surgical Margins",        "options": "negative/positive/Not mentioned",                              "gt_col": "Surgical Margins"},
        {"name": "Lymphovascular Invasion", "options": "present/absent/Not mentioned",                                 "gt_col": "Lymphovascular Invasion"},
        {"name": "Perineural Invasion",     "options": "present/absent/Not mentioned",                                 "gt_col": "Perineural Invasion"},
    ]},

    # ── Stomach Cancer (CAP / AJCC 8th ed.) ──────────────────────────────────
    "Stomach Cancer (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",         "options": "tubular adenocarcinoma/mucinous adenocarcinoma/poorly cohesive carcinoma/mixed/other/Not mentioned"},
        {"name": "Histologic Grade",        "options": "well differentiated/moderately differentiated/poorly differentiated/Not mentioned"},
        {"name": "Primary Tumor Site",      "options": "cardia/fundus/body/antrum/pylorus/gastroesophageal junction/Not mentioned"},
        {"name": "T-Stage (TNM)",           "options": "pT1a/pT1b/pT2/pT3/pT4a/pT4b/Not mentioned"},
        {"name": "N-Stage (TNM)",           "options": "pN0/pN1/pN2/pN3a/pN3b/Not mentioned"},
        {"name": "M-Stage (TNM)",           "options": "pM0/pM1/Not mentioned"},
        {"name": "Lymph Nodes Examined",    "options": "Integer/Not mentioned"},
        {"name": "Lymph Nodes Positive",    "options": "Integer/Not mentioned"},
        {"name": "Surgical Margins",        "options": "negative/positive/Not mentioned"},
        {"name": "Lymphovascular Invasion", "options": "present/absent/Not mentioned"},
        {"name": "Perineural Invasion",     "options": "present/absent/Not mentioned"},
        {"name": "Extranodal Extension",    "options": "present/absent/Not mentioned"},
        {"name": "Signet Ring Cells",       "options": "present/absent/Not mentioned"},
        {"name": "Extracellular Mucin",     "options": "present/absent/Not mentioned"},
    ]},

    # ── Stomach Cancer — TCGA-242 benchmark variant ──────────────────────────
    # All 14 fields have evaluable gold coverage in TCGA-242 (no dead fields to
    # drop); this variant only adds gt_col mappings for tcga_stomach_gt.xlsx.
    "Stomach Cancer — TCGA-242 (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",         "options": "tubular adenocarcinoma/mucinous adenocarcinoma/poorly cohesive carcinoma/mixed/other/Not mentioned", "gt_col": "Histologic Type"},
        {"name": "Histologic Grade",        "options": "well differentiated/moderately differentiated/poorly differentiated/Not mentioned", "gt_col": "Histologic Grade"},
        {"name": "Primary Tumor Site",      "options": "cardia/fundus/body/antrum/pylorus/gastroesophageal junction/Not mentioned", "gt_col": "Primary Tumor Site"},
        {"name": "T-Stage (TNM)",           "options": "pT1a/pT1b/pT2/pT3/pT4a/pT4b/Not mentioned",                    "gt_col": "T-Stage (TNM)"},
        {"name": "N-Stage (TNM)",           "options": "pN0/pN1/pN2/pN3a/pN3b/Not mentioned",                          "gt_col": "N-Stage (TNM)"},
        {"name": "M-Stage (TNM)",           "options": "pM0/pM1/Not mentioned",                                        "gt_col": "M-Stage (TNM)"},
        {"name": "Lymph Nodes Examined",    "options": "Integer/Not mentioned",                                        "gt_col": "Lymph Nodes Examined"},
        {"name": "Lymph Nodes Positive",    "options": "Integer/Not mentioned",                                        "gt_col": "Lymph Nodes Positive"},
        {"name": "Surgical Margins",        "options": "negative/positive/Not mentioned",                              "gt_col": "Surgical Margins"},
        {"name": "Lymphovascular Invasion", "options": "present/absent/Not mentioned",                                 "gt_col": "Lymphovascular Invasion"},
        {"name": "Perineural Invasion",     "options": "present/absent/Not mentioned",                                 "gt_col": "Perineural Invasion"},
        {"name": "Extranodal Extension",    "options": "present/absent/Not mentioned",                                 "gt_col": "Extranodal Extension"},
        {"name": "Signet Ring Cells",       "options": "present/absent/Not mentioned",                                 "gt_col": "Signet Ring Cells"},
        {"name": "Extracellular Mucin",     "options": "present/absent/Not mentioned",                                 "gt_col": "Extracellular Mucin"},
    ]},

    # ── Liver Cancer (CAP / AJCC 8th ed.) ────────────────────────────────────
    "Liver Cancer (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",         "options": "hepatocellular carcinoma/intrahepatic cholangiocarcinoma/combined hepatocellular-cholangiocarcinoma/other/Not mentioned"},
        {"name": "Histologic Grade",        "options": "well differentiated/moderately differentiated/poorly differentiated/undifferentiated/Not mentioned",
         "note": "If the report states more than one differentiation grade for the same tumor (e.g. a \"worst\" and a \"major\"/\"predominant\" differentiation), use the WORST (highest-grade) value. Do not use any background liver-disease activity grade (e.g. hepatitis grade/stage, \"Grade, lobular\") — that is a different, unrelated measurement of the surrounding non-tumor liver, not the tumor's own differentiation. Grades are sometimes written as Roman numerals: map them exactly as I=well differentiated, II=moderately differentiated, III=poorly differentiated, IV=undifferentiated — do not shift by one level in either direction."},
        {"name": "Tumor Size (mm)",         "options": "Number/Not mentioned"},
        {"name": "Tumor Site",              "options": "right lobe/left lobe/caudate lobe/Not mentioned"},
        {"name": "Tumor Focality",          "options": "unifocal/multifocal/Not mentioned"},
        {"name": "T-Stage (TNM)",           "options": "pT1a/pT1b/pT2/pT3/pT4/Not mentioned"},
        {"name": "N-Stage (TNM)",           "options": "pN0/pN1/Not mentioned"},
        {"name": "M-Stage (TNM)",           "options": "pM0/pM1/Not mentioned"},
        {"name": "Surgical Margins",        "options": "negative/positive/Not mentioned"},
        {"name": "Vascular Invasion",       "options": "present/absent/Not mentioned"},
        {"name": "Lymphovascular Invasion", "options": "present/absent/Not mentioned"},
    ]},

    # ── Liver Cancer — TCGA-242 benchmark variant ────────────────────────────
    # Same as "Liver Cancer (CAP/AJCC)" minus Vascular Invasion and
    # Lymphovascular Invasion, neither of which the TCGA-242 gold schema
    # records (0 evaluable cases each in tcga242_report.md §6.5) — and minus
    # T-Stage, N-Stage, and M-Stage, all three removed from scoring because
    # misses dominated their errors (≥50%) — gold computes these from criteria
    # the report text never states as an extractable token
    # (tcga242_report_summary.md, "Some T/N/M-Stage fields were removed
    # entirely"; full detail in tcga242_report.md §9.11).
    "Liver Cancer — TCGA-242 (CAP/AJCC)": {"fields": [
        {"name": "Histologic Type",         "options": "hepatocellular carcinoma/intrahepatic cholangiocarcinoma/combined hepatocellular-cholangiocarcinoma/other/Not mentioned", "gt_col": "Histologic Type"},
        {"name": "Histologic Grade",        "options": "well differentiated/moderately differentiated/poorly differentiated/undifferentiated/Not mentioned", "gt_col": "Histologic Grade",
         "note": "If the report states more than one differentiation grade for the same tumor (e.g. a \"worst\" and a \"major\"/\"predominant\" differentiation), use the WORST (highest-grade) value. Do not use any background liver-disease activity grade (e.g. hepatitis grade/stage, \"Grade, lobular\") — that is a different, unrelated measurement of the surrounding non-tumor liver, not the tumor's own differentiation. Grades are sometimes written as Roman numerals: map them exactly as I=well differentiated, II=moderately differentiated, III=poorly differentiated, IV=undifferentiated — do not shift by one level in either direction."},
        {"name": "Tumor Size (mm)",         "options": "Number/Not mentioned",                                         "gt_col": "Tumor Size (mm)"},
        {"name": "Tumor Site",              "options": "right lobe/left lobe/caudate lobe/Not mentioned",              "gt_col": "Tumor Site"},
        {"name": "Tumor Focality",          "options": "unifocal/multifocal/Not mentioned",                            "gt_col": "Tumor Focality"},
        {"name": "Surgical Margins",        "options": "negative/positive/Not mentioned",                              "gt_col": "Surgical Margins"},
    ]},

    # ── Lung Cancer NSCLC (CAP / IASLC / AJCC 8th ed.) ──────────────────────
    "Lung Cancer — NSCLC (CAP/IASLC/AJCC)": {"fields": [
        {"name": "Histologic Type",              "options": "adenocarcinoma/squamous cell carcinoma/large cell carcinoma/adenosquamous/sarcomatoid/NOS/Not mentioned"},
        {"name": "Histologic Grade",             "options": "well differentiated/moderately differentiated/poorly differentiated/undifferentiated/Not mentioned"},
        {"name": "Tumor Size (mm)",              "options": "Number/Not mentioned"},
        {"name": "T-Stage (TNM)",                "options": "pT1a/pT1b/pT1c/pT2a/pT2b/pT3/pT4/Not mentioned"},
        {"name": "N-Stage (TNM)",                "options": "pN0/pN1/pN2/pN3/Not mentioned"},
        {"name": "Resection Margins",            "options": "R0/R1/R2/Not mentioned"},
        {"name": "Visceral Pleural Invasion",    "options": "PL0/PL1/PL2/PL3/Not mentioned"},
        {"name": "Lymphovascular Invasion",      "options": "present/absent/Not mentioned"},
        {"name": "Perineural Invasion",          "options": "present/absent/Not mentioned"},
        {"name": "EGFR Mutation",                "options": "mutated/wild-type/not tested/Not mentioned"},
        {"name": "ALK Status",                   "options": "positive/negative/not tested/Not mentioned"},
        {"name": "PD-L1 Expression (TPS %)",     "options": "Number/not tested/Not mentioned"},
        {"name": "KRAS Mutation",                "options": "mutated/wild-type/not tested/Not mentioned"},
    ]},

    # ── Tongue / Oral Cavity SCC ──────────────────────────────────────────────
    "Tongue / Oral SCC (Head & Neck)": {"fields": [
        {"name": "Tongue Involved",         "options": "yes/no/unclear"},
        {"name": "Biopsy",                  "options": "yes/no"},
        {"name": "Resection",               "options": "yes/no"},
        {"name": "SCC Confirmed",           "options": "yes/no/in_situ_only/unclear"},
        {"name": "Histologic Grade",        "options": "well differentiated/moderately differentiated/poorly differentiated/undifferentiated/Not mentioned"},
        {"name": "Tumor Size (cm)",         "options": "Number/Not mentioned"},
        {"name": "Depth of Invasion (mm)",  "options": "Number/Not mentioned"},
        {"name": "Margin Status",           "options": "negative/positive/close/not assessed/Not mentioned"},
        {"name": "Lymphovascular Invasion", "options": "present/absent/Not mentioned"},
        {"name": "Perineural Invasion",     "options": "present/absent/Not mentioned"},
        {"name": "Worst Pattern of Invasion","options": "1/2/3/4C/5/Not mentioned"},
        {"name": "pT Stage",                "options": "pT1/pT2/pT3/pT4a/pT4b/Not mentioned"},
        {"name": "pN Stage",                "options": "pN0/pN1/pN2a/pN2b/pN2c/pN3a/pN3b/Not mentioned"},
        {"name": "HPV/p16 Status",          "options": "positive/negative/not tested/Not mentioned"},
    ]},

    # ── Tongue Cancer — GT-matched (matches tongue_gt.xlsx columns exactly) ──────
    "Tongue Cancer — GT matched": {"fields": [
        {"name": "Tongue Involved", "options": "yes/no/Not mentioned",              "gt_col": "Tongue Involved"},
        {"name": "Biopsy",          "options": "yes/no/Not mentioned",              "gt_col": "Biopsy"},
        {"name": "Resection",       "options": "yes/no/Not mentioned",              "gt_col": "Resection"},
        {"name": "SCC Confirmed",   "options": "yes/no/in_situ_only/unclear/Not mentioned", "gt_col": "SCC Confirmed"},
        {"name": "pT Stage",        "options": "String/Not mentioned",              "gt_col": "pT Stage"},
    ]},

    "Custom (define below)": {"fields": []},
}

# Keyword -> preset guesses for auto-selecting the Preset dropdown from an
# uploaded report filename. Each cancer type lists (keywords, plain preset,
# TCGA-242 preset or None) — the TCGA-242 variant is picked instead when the
# filename also contains "tcga".
FILENAME_PRESET_HINTS = [
    (["prostate", "prad"],                          "Prostate Cancer — PRAD (Grothey et al. 2025)", None),
    (["thyroid"],                                   "Thyroid Cancer (Lee et al. 2025)", None),
    (["breast"],                                    "Breast Cancer (CAP/AJCC)", "Breast Cancer — TCGA-242 (CAP/AJCC)"),
    (["colorectal", "colon", "rectal", "rectum"],   "Colorectal Cancer (CAP/AJCC)", "Colorectal Cancer — TCGA-242 (CAP/AJCC)"),
    (["esophagus", "esophageal", "oesophagus"],     "Esophagus Cancer (CAP/AJCC)", "Esophagus Cancer — TCGA-242 (CAP/AJCC)"),
    (["stomach", "gastric"],                        "Stomach Cancer (CAP/AJCC)", "Stomach Cancer — TCGA-242 (CAP/AJCC)"),
    (["liver", "hepatocellular", "hcc"],            "Liver Cancer (CAP/AJCC)", "Liver Cancer — TCGA-242 (CAP/AJCC)"),
    (["lung", "nsclc"],                             "Lung Cancer — NSCLC (CAP/IASLC/AJCC)", None),
    (["tongue", "oral"],                            "Tongue / Oral SCC (Head & Neck)", None),
]

def guess_preset_from_filename(filename: str):
    """Best-effort preset guess from an uploaded filename, or None if no keyword matches."""
    name = filename.lower()
    is_tcga = "tcga" in name
    for keywords, plain_preset, tcga_preset in FILENAME_PRESET_HINTS:
        if any(kw in name for kw in keywords):
            return tcga_preset if (is_tcga and tcga_preset) else plain_preset
    return None

NAV_PAGES = [
    "📋 Fields & Prompt",
    "▶️ Run & Results",
    "📊 Compare Strategies",
    "✏️ Review",
    "🎯 Cohort",
    "❓ Help",
]

STRATEGIES = {
    "cot":              "Chain-of-Thought",
    "cov":              "Chain-of-Verif.",
    "persona_cot":      "Persona · CoT",
    "persona_cov":      "Persona · CoV",
    "persona_fewshot":  "Persona · Few-shot",
    "persona_strict":   "Persona · Strict",
    "persona_zeroshot": "Persona · Zero-shot",
    "zero_shot":        "Zero-shot",
}

STRATEGY_HINTS = {
    "cot":              "Step-by-step reasoning before JSON.",
    "cov":              "Two API calls: extract → verify.",
    "persona_cot":      "Expert persona + reasoning embedded in JSON.",
    "persona_cov":      "Expert persona + two-pass verification.",
    "persona_fewshot":  "Expert persona + one worked example.",
    "persona_strict":   "Expert persona + 5 strict rules.",
    "persona_zeroshot": "Expert pathologist persona + standard extraction.",
    "zero_shot":        "Paper-exact prompt, no persona.",
}

PERSONA = (
    "You are a highly experienced pathologist with over 20 years of expertise "
    "in diagnosing and reporting on various types of cancers. Your task is to "
    "extract specific information from pathology reports with precision and accuracy.\n\n"
)

FEW_SHOT_EXAMPLE = {
    "report": (
        "Prostatectomy specimen (52 g). Bilateral infiltrates of acinar adenocarcinoma "
        "with a Gleason score of 7 (4+3). Extracapsular extension on the right (pT3a). "
        "Resection margins free (R0). 0/12 lymph nodes positive (pN0)."
    ),
    "output": {
        "WHO (ISUP) Grade Group": "GG3", "T-Stage (TNM)": "pT3a", "N-Stage (TNM)": "pN0",
        "Number of Lymph Nodes examined": "12", "Lymph Nodes with Metastasis": "0",
        "Resection Margins": "R0", "Histologic Subtype": "acinar",
        "Primary Gleason Pattern": "4", "Secondary Gleason Pattern": "3",
        "Tertiary Gleason Pattern": "Not mentioned",
        "Percentage of Secondary Gleason Pattern": "Not mentioned",
    },
}

# ── Prompt builders ────────────────────────────────────────────────────────────

def _lines(fields):
    out = []
    for i, f in enumerate(fields):
        line = f"{i+1}.\t{f['name']}: [{f['options']}]"
        if f.get("note"):
            line += f" ({f['note']})"
        out.append(line)
    return "\n".join(out)

def _base(fields):
    return (
        "Please extract key information from the provided pathology report and "
        "categorize it under the respective headings. Listed below are the parameters "
        "that should be retrieved from the report. All of them are formatted as follows:\n\n"
        "    Heading: [Possible options]\n\n"
        "Stick precisely to the given answer options for each category. "
        "If an answer option specifies 'Integer' or 'Number', ensure the response is "
        "provided as a numeric value. If the report does not specify information about "
        "a particular category or if the information is unclear, provide the response "
        "as \"Not mentioned.\"\n\n"
        "The output should be structured in a JSON format.\n\n"
        + _lines(fields) + "\n\nThe report reads:"
    )

def _strict(fields):
    return (
        PERSONA
        + "Extract the following fields. Follow these rules strictly:\n"
        "1. Use ONLY the provided answer options — no synonyms, no abbreviations.\n"
        "2. If a field is not explicitly stated, respond with \"Not mentioned\".\n"
        "3. For Integer/Number fields, respond with numerals only.\n"
        "4. Do not infer or assume values not clearly stated in the report.\n"
        "5. Return a valid JSON object with exactly these keys.\n\n"
        + _lines(fields) + "\n\nThe report reads:"
    )

def _cot(fields):
    return (
        PERSONA
        + "Extract the following fields from the pathology report. "
        "Reason step by step for each field, then return a JSON object "
        "with a \"thinking\" key (your reasoning) followed by the field keys.\n\n"
        + _lines(fields) + "\n\nThe report reads:"
    )

def _fewshot(fields):
    prad_names = {f["name"] for f in
                  PRESETS["Prostate Cancer — PRAD (Grothey et al. 2025)"]["fields"]}
    current_names = {f["name"] for f in fields}

    if current_names == prad_names:
        ex = FEW_SHOT_EXAMPLE
        example_block = (
            f"Example:\nReport: {ex['report']}\n"
            f"Output:\n{json.dumps(ex['output'], indent=2)}"
        )
    else:
        # A field's first listed option is only a usable example value for
        # categorical fields (e.g. "present/absent/Not mentioned" -> "present").
        # For Number/Integer/String fields, the first option is the literal
        # placeholder word itself (e.g. "Number/Not mentioned" -> "Number"),
        # which would show the model a fake example that contradicts the
        # prompt's own instruction to answer with a real numeric value.
        PLACEHOLDER_EXAMPLES = {"Number": "12.5", "Integer": "3", "String": "Not mentioned"}
        def _example_value(options):
            first = options.split("/")[0].strip()
            return PLACEHOLDER_EXAMPLES.get(first, first)
        dynamic_output = {f["name"]: _example_value(f["options"]) for f in fields}
        example_block = (
            "Example output format (illustrative values only — replace with "
            "the value actually found in the report):\n"
            + json.dumps(dynamic_output, indent=2)
        )

    return (
        PERSONA + "Extract the following fields from pathology reports.\n\n"
        + _lines(fields)
        + f"\n\n{example_block}"
        "\n\n---\nNow extract from the report below. Return JSON only.\n\nThe report reads:"
    )

def _cov_verify(fields, persona=False):
    prefix = PERSONA if persona else ""
    return (
        prefix
        + "Given an original pathology report and a JSON extraction, verify accuracy "
        "and completeness for each category below. Correct any discrepancies.\n\n"
        + _lines(fields)
        + "\n\nReturn only the corrected JSON object."
    )

def build_prompt(fields, strategy):
    return {
        "zero_shot":        _base(fields),
        "cot":              _base(fields).replace("\n\nThe report reads:", "")
                            + "\n\nWalk through your answer step by step, then summarise in JSON.\n\nThe report reads:",
        "cov":              _base(fields),
        "persona_zeroshot": PERSONA + _base(fields),
        "persona_strict":   _strict(fields),
        "persona_cot":      _cot(fields),
        "persona_cov":      PERSONA + _base(fields),
        "persona_fewshot":  _fewshot(fields),
    }.get(strategy, _base(fields))

def cov_system(fields, strategy):
    return _cov_verify(fields, persona=(strategy == "persona_cov"))

# ── Utilities ──────────────────────────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("_x000D_\n", "\n").replace("_x000D_", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()

def cache_key(text, prompt, model="", url=""):
    # model/url must be part of the key — otherwise comparing multiple
    # models (or providers) on the same report+prompt silently reuses
    # whichever model's response was cached first for every other model,
    # since text+prompt alone can be identical across models.
    return hashlib.sha256(f"{text}|||{prompt}|||{model}|||{url}".encode()).hexdigest()

def _cache_file(username):
    """Each signed-in user gets their own cache file — never shared with,
    or readable by, any other user. Signed-out (anonymous) browsing has no
    username and therefore no persistent cache at all."""
    if not username:
        return None
    return user_dir(username) / "extraction_cache.csv"

def load_cache(username):
    cf = _cache_file(username)
    if cf is not None and cf.exists():
        try:
            return pd.read_csv(cf, dtype=str)
        except Exception:
            pass
    return pd.DataFrame(columns=["cache_key", "json_result", "created_at"])

def append_cache(username, ck, json_result):
    cf = _cache_file(username)
    if cf is None:   # no signed-in user — nothing to persist to
        return
    row = pd.DataFrame([{"cache_key": ck, "json_result": json_result,
                          "created_at": datetime.utcnow().isoformat()}])
    row.to_csv(cf, mode="a", header=not cf.exists(), index=False)

# Runtime-discovered models that rejected temperature (populated on first failure)
_no_temp_discovered: set = set()

def supports_temperature(model: str) -> bool:
    """Return False for models known not to accept a temperature parameter."""
    base = model.lower().split("/")[-1]
    if base in _no_temp_discovered:
        return False
    return not any(base == p or base.startswith(p + "-") or base.startswith(p + ".")
                   for p in _NO_TEMP_PREFIXES)

def _is_temp_error(response_text: str) -> bool:
    t = response_text.lower()
    return "temperature" in t and any(w in t for w in
                                      ("not supported", "unsupported", "invalid",
                                       "does not support", "cannot be"))

def call_api(text, system_prompt, url, key, model, max_tokens=8000,
             json_mode=True, temperature=0.0):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": text or "<missing>"}],
        "max_completion_tokens": int(max_tokens),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if supports_temperature(model):
        body["temperature"] = float(temperature)
    try:
        r = requests.post(url, headers=headers, json=body, timeout=120)
        # If temperature caused a 400, retry without it and remember for this session
        if r.status_code == 400 and "temperature" in body and _is_temp_error(r.text):
            _no_temp_discovered.add(model.lower().split("/")[-1])
            body.pop("temperature")
            r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        choices = r.json().get("choices", [])
        if not choices:
            return {"message": ""}
        choice  = choices[0]
        content = choice["message"].get("content") or ""
        # Some models (e.g. Gemini's reasoning models) spend a variable,
        # sometimes large number of completion tokens on internal "thinking"
        # before writing the visible answer — that counts against
        # max_completion_tokens. When the budget runs out mid-answer, the
        # API still returns 200 with finish_reason="length" and partial (or
        # empty) JSON, which would otherwise fail parse_json() silently and
        # show up as "None" in every field with no visible error. Surface it
        # as an explicit error instead so the run log actually explains it.
        if choice.get("finish_reason") == "length":
            return {"error": f"Response truncated at max_tokens={max_tokens} "
                              f"before finishing — raise Max tokens (try 2000+) and retry."}
        return {"message": content}
    except Exception as e:
        return {"error": str(e)}

def parse_json(message, field_names):
    if not message:
        return {f: None for f in field_names}
    text = message.strip()
    parsed = None

    # 1. Whole-message JSON (typical for OpenAI/Gemini with json_object mode).
    try:
        parsed = json.loads(text)
    except Exception:
        pass

    # 2. A ```json ... ``` fenced block anywhere in the text — some models
    # (e.g. Claude) append explanatory prose *after* the closing fence,
    # which a start/end-anchored strip can't handle.
    if parsed is None:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                pass

    # 3. Fall back to the first balanced {...} object anywhere in the text.
    if parsed is None:
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
                            parsed = json.loads(text[start:i + 1])
                        except Exception:
                            pass
                        break

    if parsed is None:
        return {f: None for f in field_names}
    out = {}
    for f in field_names:
        v = parsed.get(f)
        out[f] = str(v[0] if isinstance(v, list) else v) if v is not None else None
    return out

def norm_gt(val, rule=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "not mentioned"
    s = str(val).strip()
    if rule == "integer":
        try: return str(int(float(s)))
        except: return s.lower()
    if rule == "pct":
        try: return str(int(float(s) * 100))
        except: return s.lower()
    if rule == "lym_pos":
        return s.split("/")[0].strip().lower()
    if rule == "lym_total":
        return (s.split("/")[-1] if "/" in s else s).strip().lower()
    if rule == "gp3":
        v = s.split(",")[0].strip()
        return "not mentioned" if v.lower() in ("na", "nan", "", "not mentioned") else v.lower()
    return s.lower()

# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_one(note, prompt, strategy, fields, url, key, model,
                max_tokens, disk_cache, username, temperature=0.0, use_cache=True):
    is_cov = strategy in ("cov", "persona_cov")
    is_cot = strategy == "cot"
    ck = cache_key(note, prompt, model, url)

    if use_cache and disk_cache is not None and ck in disk_cache["cache_key"].values:
        # .iloc[-1]: if a case was force-re-run before, its newest entry — not
        # its original — is the one that should count as "the cached answer".
        cached = disk_cache.loc[disk_cache["cache_key"] == ck, "json_result"].iloc[-1]
        return {"json": cached, "source": "disk", "n_calls": 0}

    res = call_api(note, prompt, url, key, model, max_tokens,
                   json_mode=not is_cot, temperature=temperature)
    if "error" in res:
        return {"json": None, "source": "error", "error": res["error"], "n_calls": 1}

    if is_cov:
        cov_msg = (f"The report reads:\n{note}\n"
                   f"The model output reads:\n{res['message']}\n"
                   "Return the corrected JSON object.")
        res2 = call_api(cov_msg, cov_system(fields, strategy), url, key, model,
                        max_tokens, temperature=temperature)
        if "error" not in res2:
            res = res2
        n_calls = 2
    else:
        n_calls = 1

    append_cache(username, ck, res["message"])
    return {"json": res["message"], "source": "fresh", "n_calls": n_calls}

def collapse_subreports(rows, field_names):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    def coalesce(s):
        # Most recent sub-report wins (last row in group = highest Sub_ID)
        valid = [v for v in reversed(list(s))
                 if v and str(v).lower() not in ("not mentioned", "none", "nan", "")]
        return valid[0] if valid else next((v for v in s if pd.notna(v)), None)
    agg = {f: coalesce for f in field_names if f in df.columns}
    return df.groupby("Case_ID", sort=False).agg(agg).reset_index()

def compute_accuracy(results_df, gt_df, fields, gt_id_col, gt_col_map):
    gt = gt_df.rename(columns={gt_id_col: "Case_ID"})

    # Rename GT columns to avoid suffix collision when field names == GT column names
    gt_rename = {c: c + "__gt" for c in gt.columns if c != "Case_ID"}
    gt = gt.rename(columns=gt_rename)

    merged = results_df.merge(gt, on="Case_ID")
    rows = []
    for f in fields:
        fname        = f["name"]
        gt_col_orig  = gt_col_map.get(fname, "")
        gt_norm_rule = f.get("gt_norm", "")
        if not gt_col_orig:
            continue
        gt_col_merged = gt_col_orig + "__gt"   # name after our rename
        if fname not in merged.columns or gt_col_merged not in merged.columns:
            continue
        pred  = merged[fname].fillna("not mentioned").astype(str).str.lower().str.strip()
        gold  = merged[gt_col_merged].apply(lambda v: norm_gt(v, gt_norm_rule))
        valid = ~gold.isin(["9", "na", ""])
        n     = int(valid.sum())
        if n == 0:
            continue
        acc = round((pred[valid] == gold[valid]).mean() * 100, 1)
        rows.append({"Field": fname, "N": n, "Accuracy (%)": acc})
    return pd.DataFrame(rows) if rows else None

def compute_classification_metrics(results_df, gt_df, fields, gt_id_col, gt_col_map, positive_class):
    """For fields with a user-configured positive class, compute sensitivity,
    specificity, PPV, and NPV against ground truth.

    `positive_class` maps field name -> either a list/set of option strings
    (categorical positive class, e.g. {"HER2 Status": ["Positive"]}) or the
    literal string ">0" for numeric fields, treating any value greater than
    zero as positive (e.g. a lymph-node-count field where >0 means involved).
    Fields absent from `positive_class` are skipped entirely.
    """
    if not positive_class:
        return {}
    gt = gt_df.rename(columns={gt_id_col: "Case_ID"})
    gt_rename = {c: c + "__gt" for c in gt.columns if c != "Case_ID"}
    gt = gt.rename(columns=gt_rename)
    merged = results_df.merge(gt, on="Case_ID")

    out = {}
    for f in fields:
        fname = f["name"]
        if fname not in positive_class:
            continue
        gt_col_orig  = gt_col_map.get(fname, "")
        gt_norm_rule = f.get("gt_norm", "")
        if not gt_col_orig:
            continue
        gt_col_merged = gt_col_orig + "__gt"
        if fname not in merged.columns or gt_col_merged not in merged.columns:
            continue
        pred  = merged[fname].fillna("not mentioned").astype(str).str.lower().str.strip()
        gold  = merged[gt_col_merged].apply(lambda v: norm_gt(v, gt_norm_rule))
        valid = ~gold.isin(["9", "na", ""])
        pred, gold = pred[valid], gold[valid]
        if len(pred) == 0:
            continue

        spec = positive_class[fname]
        if spec == ">0":
            pred_pos = pd.to_numeric(pred, errors="coerce").fillna(0) > 0
            gold_pos = pd.to_numeric(gold, errors="coerce").fillna(0) > 0
        else:
            pos_vals = {str(v).lower().strip() for v in spec}
            pred_pos = pred.isin(pos_vals)
            gold_pos = gold.isin(pos_vals)

        tp = int((pred_pos & gold_pos).sum())
        fn = int((~pred_pos & gold_pos).sum())
        fp = int((pred_pos & ~gold_pos).sum())
        tn = int((~pred_pos & ~gold_pos).sum())

        sens = round(100 * tp / (tp + fn), 1) if (tp + fn) > 0 else None
        spc  = round(100 * tn / (tn + fp), 1) if (tn + fp) > 0 else None
        ppv  = round(100 * tp / (tp + fp), 1) if (tp + fp) > 0 else None
        npv  = round(100 * tn / (tn + fn), 1) if (tn + fn) > 0 else None

        out[fname] = {
            "Sensitivity (%)": sens, "Specificity (%)": spc,
            "PPV (%)": ppv, "NPV (%)": npv,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        }
    return out

def apply_corrections(results_df, corrections):
    """Return a copy of results_df with manual corrections applied."""
    if results_df is None:
        return None
    df = results_df.copy()
    for cid, field_dict in corrections.items():
        mask = df["Case_ID"] == cid
        for field, val in field_dict.items():
            if field in df.columns and str(val).strip().lower() not in ("", "nan", "none"):
                df.loc[mask, field] = str(val).strip()
    return df

# ── Cohort helpers ─────────────────────────────────────────────────────────────

OPERATORS = ["==", "!=", "contains", "not contains",
             "in (comma list)", "not in (comma list)",
             "is missing", "is present",
             ">", ">=", "<", "<="]

def _to_float_series(series):
    """Best-effort numeric parse; non-numeric cells (including "Not mentioned")
    become NaN and so never match a numeric comparison, rather than raising."""
    return pd.to_numeric(
        series.astype(str).str.strip().str.rstrip("%"), errors="coerce")

def eval_condition(series, op, value):
    s = series.fillna("").astype(str).str.lower().str.strip()
    v = str(value).lower().strip()
    if op == "==":            return s == v
    if op == "!=":            return s != v
    if op == "contains":      return s.str.contains(v, regex=False, na=False)
    if op == "not contains":  return ~s.str.contains(v, regex=False, na=False)
    if op == "in (comma list)":
        vals = {x.strip().lower() for x in v.split(",") if x.strip()}
        return s.isin(vals)
    if op == "not in (comma list)":
        vals = {x.strip().lower() for x in v.split(",") if x.strip()}
        return ~s.isin(vals)
    if op == "is missing":    return s.isin(["", "not mentioned", "nan", "none"])
    if op == "is present":    return ~s.isin(["", "not mentioned", "nan", "none"])
    if op in (">", ">=", "<", "<="):
        num_series = _to_float_series(series)
        try:
            num_value = float(str(value).strip().rstrip("%"))
        except ValueError:
            return pd.Series([False] * len(series), index=series.index)
        if op == ">":  return num_series >  num_value
        if op == ">=": return num_series >= num_value
        if op == "<":  return num_series <  num_value
        if op == "<=": return num_series <= num_value
    return pd.Series([False] * len(series), index=series.index)

def build_cohort(results_df, corrections, case_cond, ctrl_cond, ie_rules):
    df = apply_corrections(results_df, corrections).copy()

    # Case / Control
    df["Group"] = "Unclassified"
    for label, cond in (("Case", case_cond), ("Control", ctrl_cond)):
        f, op, val = cond.get("field",""), cond.get("op","=="), cond.get("value","")
        if f and f in df.columns and val:
            df.loc[eval_condition(df[f], op, val), "Group"] = label

    # Inclusion / Exclusion
    df["Cohort"] = "Included"
    df["Exclusion reason"] = ""
    for rule in ie_rules:
        f   = rule.get("Field", "")
        op  = rule.get("Operator", "==")
        val = str(rule.get("Value", "")).strip()
        act = rule.get("Action", "Exclude if")
        if not f or f not in df.columns or not val:
            continue
        mask    = eval_condition(df[f], op, val)
        still_in = df["Cohort"] == "Included"
        if act == "Exclude if":
            hit = mask & still_in
        else:                                          # Include only if
            hit = ~mask & still_in
        reason = f"{f} {op} {val}" if act == "Exclude if" else f"NOT ({f} {op} {val})"
        df.loc[hit, "Cohort"] = "Excluded"
        df.loc[hit, "Exclusion reason"] = reason

    return df

def build_data_quality_heatmap(df, field_names):
    """Return (z, text, hover, case_ids) for a value-annotated quality heatmap.
    z      : 0=missing, 1=not mentioned, 2=extracted  (drives cell colour)
    text   : short label shown inside each cell
    hover  : full value shown on mouse-over
    """
    z, text, hover = [], [], []
    for _, row in df.iterrows():
        zr, tr, hr = [], [], []
        for fname in field_names:
            raw = row[fname] if fname in row.index else None
            val = str(raw).strip() if raw is not None else ""
            low = val.lower()
            if low in ("", "nan", "none"):
                zr.append(0); tr.append("—");    hr.append(f"{fname}: missing")
            elif low == "not mentioned":
                zr.append(1); tr.append("n/m");  hr.append(f"{fname}: not mentioned")
            else:
                zr.append(2)
                short = val if len(val) <= 6 else val[:5] + "…"
                tr.append(short); hr.append(f"{fname}: {val}")
        z.append(zr); text.append(tr); hover.append(hr)
    return z, text, hover, df["Case_ID"].tolist()

# ── Session state ──────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        # data
        "report_df": None, "gt_df": None,
        "results_df": None, "corrections": {}, "compare_df": None,
        "positive_class": {},  # field name -> [positive option(s)] or ">0", for sens/spec/PPV/NPV
        "cohort_df": None,
        "case_cond":  {"field": "", "op": "==", "value": ""},
        "ctrl_cond":  {"field": "", "op": "==", "value": ""},
        "ie_rules":   [],
        "ie_rules_version": 0,  # bumped when nl_cohort programmatically adds rules,
                                # so the data_editor re-seeds instead of showing stale state
        "nl_cohort_proposal": None,  # last NL-to-rules translation, awaiting review
        "case_explanations": {},  # case_id -> {field: {"text","quote","span"}}, Review tab
        "review_highlight": None,  # (case_id, field) currently highlighted in the report pane
        "_save_msg": "",   # persists last save confirmation across reruns
        "_save_count": 0,  # incremented on save to reset the data_editor key
        "_gt_mapping_seen": False,  # forces the GT-mapping expander's initial
                                    # open/closed state exactly once; after
                                    # that its state is left alone so mapping
                                    # one field doesn't re-collapse it on
                                    # every subsequent field
        # fields & strategy (persist across tabs)
        "fields": [], "fields_version": 0,
        "preset_choice": list(PRESETS.keys())[0],
        "strategy": "persona_zeroshot", "max_tokens": 8000,
        "prompt_override": False, "custom_prompt": "",
        "saved_prompts": {},   # name -> prompt text; per-user, reloaded on login
        # column selections
        "id_col": None, "text_col": None,
        "gt_id_col": None, "gt_col_map": {},
        # API
        "api_key": "", "_api_key_draft": "", "api_url": SHARED_URL,
        "azure_resource": "",
        "api_model": SHARED_MODEL, "key_mode": "Shared key" if SHARED_ENABLED else "My own key",
        "temperature": 0.0, "_last_provider": "OpenAI",
        "session_calls": 0,
        "live_models": {},        # provider -> list[str] fetched from its API, session-only
        "live_models_error": {},  # provider -> str, last fetch's error message (if any)
        # compare
        "compare_selected": ["persona_zeroshot", "persona_strict", "persona_cot"],
        "model_compare_selected": [],  # list of "Provider::Model" ids, like compare_selected
        "model_compare_keys":  {},    # {provider: api_key}, session-only, never persisted
        "model_compare_df":   None,
        "model_compare_strategy": "persona_zeroshot",
        # review navigation
        "review_idx": 0,
        # guided page navigation (replaces manual tab-clicking)
        "nav_page": 0,
        "nav_pill": NAV_PAGES[0],
        # cache
        "disk_cache": load_cache(None),   # anonymous by default; reloaded per-user on login
        # file tracking
        "_report_fid": None, "_gt_fid": None,
        # PHI pre-pass scan result cache, keyed by (report_fid, id_col, text_col)
        # so the regex scan doesn't rerun on every Streamlit script rerun
        "_phi_scan_key": None, "_phi_scan_hits": None,
        # bumped on sign-out so the file_uploader widgets remount with no
        # file attached — clearing session_state alone doesn't reset their
        # client-side visual state, since that's tied to the widget key.
        "uploader_epoch": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
        st.session_state.setdefault("_state_loaded", False)

# ── Per-user persistence ───────────────────────────────────────────────────────

_PERSIST_KEYS = [
    ("fields",        "fields",        []),
    ("strategy",      "strategy",      "persona_zeroshot"),
    ("max_tokens",    "max_tokens",    8000),
    ("temperature",   "temperature",   0.0),
    ("api_url",       "api_url",       SHARED_URL),
    ("azure_resource", "azure_resource", ""),
    ("api_model",     "api_model",     SHARED_MODEL),
    ("_last_provider", "last_provider", "OpenAI"),   # keep in sync with api_url/api_model on login
    ("corrections",   "corrections",   {}),
    ("case_cond",     "case_cond",     {"field":"","op":"==","value":""}),
    ("ctrl_cond",     "ctrl_cond",     {"field":"","op":"==","value":""}),
    ("ie_rules",      "ie_rules",      []),
]

def _load_user_session(username: str, ss):
    """Restore saved settings + corrections from disk into session state."""
    for ss_key, file_key, default in _PERSIST_KEYS:
        saved = load_user_state(username, file_key, default)
        if saved is not None:
            ss[ss_key] = saved
    # Restore last results DataFrame
    df = load_user_csv(username, "results")
    if df is not None:
        ss.results_df = df
    # Each user's extraction cache is private to them — load it fresh on
    # login rather than carrying over whatever was in session_state from
    # anonymous browsing (or, in principle, another account).
    ss.disk_cache = load_cache(username)
    # Named custom prompts are per-user, same reasoning as the cache above.
    ss.saved_prompts = load_user_state(username, "saved_prompts", {})
    # API keys aren't persisted by default, but between sign-out and sign-in
    # the app is briefly browsable while logged out, during which the
    # sidebar's default-provider logic can auto-set ss.api_key to the shared
    # key. If the account being logged into was last using a different
    # provider, the provider-changed check won't fire this run (the
    # just-loaded _last_provider already matches what the sidebar renders),
    # so that leftover key would otherwise survive into the new session.
    # Force a clean slate, then restore an explicitly-saved secret over it
    # if the user opted in to "remember this" on a previous visit.
    ss.api_key        = ""
    ss._api_key_draft = ""
    saved_secret = load_user_secret(username)
    if saved_secret:
        ss._last_provider  = saved_secret.get("provider", ss._last_provider)
        ss.api_key         = saved_secret.get("api_key", "")
        ss.api_url         = saved_secret.get("api_url", ss.api_url)
        ss.api_model       = saved_secret.get("api_model", ss.api_model)
        ss.azure_resource  = saved_secret.get("azure_resource", ss.azure_resource)
        ss.key_mode        = "My own key"

def _save_user_session(username: str, ss):
    """Flush current settings + corrections to disk."""
    for ss_key, file_key, _ in _PERSIST_KEYS:
        val = ss.get(ss_key)
        if val is not None:
            save_user_state(username, file_key, val)
    if ss.get("results_df") is not None:
        save_user_csv(username, "results", ss.results_df)

@st.cache_data
def _load_user_guide_html() -> str:
    path = Path(__file__).parent / "user_guide.html"
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _render_report_pane(report_text: str, active_span: tuple[int, int] | None,
                         key_prefix: str, height: str = "78vh") -> None:
    """
    Read-only report pane for the Review tab: a search box (all matches
    shown, current match emphasized, ◀▶ to step through) plus the div that
    displays report_text. Search and the field-level 📍 highlight
    (active_span, from evidence_reasoning's located quote) don't overlay --
    a non-empty search takes over the pane; active_span shows only when
    the search box is empty. key_prefix must be unique per pane instance
    rendered in the same script run (the inline pane and the "Expand"
    dialog each need their own widget keys, since Streamlit can't have two
    widgets sharing one key on screen at once).
    """
    search_key = f"rsearch_{key_prefix}"
    idx_key    = f"rsearch_idx_{key_prefix}"
    lastq_key  = f"rsearch_lastq_{key_prefix}"

    sc1, sc2, sc3, sc4 = st.columns([6, 1, 1, 2])
    with sc1:
        query = st.text_input(
            "Find in report", key=search_key, placeholder="🔍 find in report…",
            label_visibility="collapsed",
        ).strip()

    if st.session_state.get(lastq_key) != query:
        st.session_state[idx_key] = 0
        st.session_state[lastq_key] = query

    matches = ([(m.start(), m.end()) for m in re.finditer(re.escape(query), report_text, re.IGNORECASE)]
               if query else [])
    n_matches = len(matches)
    cur_idx   = st.session_state.get(idx_key, 0) % n_matches if n_matches else 0

    with sc2:
        if st.button("◀", key=f"{search_key}_prev", disabled=n_matches == 0,
                      use_container_width=True, help="Previous match"):
            st.session_state[idx_key] = (cur_idx - 1) % n_matches
            st.rerun()
    with sc3:
        if st.button("▶", key=f"{search_key}_next", disabled=n_matches == 0,
                      use_container_width=True, help="Next match"):
            st.session_state[idx_key] = (cur_idx + 1) % n_matches
            st.rerun()
    with sc4:
        if query:
            st.caption(f"{cur_idx + 1} / {n_matches}" if n_matches else "no matches")

    scroll_target = None
    if matches:
        chunks, pos = [], 0
        for i, (start, end) in enumerate(matches):
            current = (i == cur_idx)
            chunks.append(html.escape(report_text[pos:start]))
            style = "background:#ff9d3b; color:#1a1a1a;" if current else "background:#ffe58a; color:#1a1a1a;"
            id_attr = ' id="search-current"' if current else ""
            chunks.append(
                f'<mark{id_attr} style="{style} padding:0 2px; border-radius:3px; '
                f'scroll-margin:2rem;">{html.escape(report_text[start:end])}</mark>'
            )
            pos = end
        chunks.append(html.escape(report_text[pos:]))
        body_html = "".join(chunks)
        scroll_target = "search-current"
    elif active_span:
        start, end = active_span
        body_html = (
            html.escape(report_text[:start])
            + '<mark id="hl-target" style="background:#ffe58a; color:#1a1a1a; '
              'padding:0 2px; border-radius:3px; scroll-margin:2rem;">'
            + html.escape(report_text[start:end]) + "</mark>"
            + html.escape(report_text[end:])
        )
        scroll_target = "hl-target"
    else:
        body_html = html.escape(report_text)

    # A disabled st.text_area renders with Streamlit's greyed-out "inactive"
    # styling, which hurts readability for something meant to be read
    # closely (not edited). A plain styled div keeps full-contrast,
    # theme-matched text while staying visually read-only (bordered,
    # scrollable, no input caret).
    st.markdown(
        f"""<div style="
                height: {height}; overflow-y: auto; padding: 0.75rem 1rem;
                border: 1px solid rgba(128,128,128,0.4); border-radius: 0.5rem;
                background-color: var(--secondary-background-color);
                color: var(--text-color);
                white-space: pre-wrap; word-wrap: break-word;
                font-family: inherit; font-size: 0.9rem; line-height: 1.5;
            ">{body_html}</div>""",
        unsafe_allow_html=True,
    )
    if scroll_target:
        # st.markdown's unsafe_allow_html renders straight into the main
        # app DOM (not a nested frame), so the mark above is a sibling of
        # whatever this component.html iframe's parent document is --
        # window.parent.document reaches it.
        components.html(
            f"""<script>
            setTimeout(function () {{
                var el = window.parent.document.getElementById('{scroll_target}');
                if (el) {{ el.scrollIntoView({{block: 'center', behavior: 'smooth'}}); }}
            }}, 50);
            </script>""",
            height=0,
        )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Pathology Structured Extractor",
        page_icon="🔬", layout="wide",
        initial_sidebar_state="expanded",
    )

    # Bigger, easier-to-hit pills for the main page-navigation bar (the
    # default segmented_control sizing reads as cramped at this page's
    # information density).
    #
    # Background tint: nearly all body text (report text, extracted-field
    # lists, Help copy) sits directly on this background rather than inside a
    # bordered card, and Streamlit's own theme (light or dark, user-toggled
    # via Settings) sets its background-color on the *parent* .stApp element,
    # not this one -- so this must stay a translucent overlay (rgba, no
    # opaque background-color) rather than a solid gradient, or it paints
    # over the dark theme's background and makes text unreadable in it.
    st.markdown("""
        <style>
        button[data-testid^="stBaseButton-segmented_control"] {
            font-size: 1.05rem !important;
            padding: 0.7rem 1.3rem !important;
            height: auto !important;
        }
        button[data-testid^="stBaseButton-segmented_control"] p {
            font-size: 1.05rem !important;
        }
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(160deg,
                rgba(90, 59, 148, 0.055) 0%,
                rgba(192, 87, 122, 0.045) 50%,
                rgba(90, 59, 148, 0.05) 100%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Authentication ─────────────────────────────────────────────────────────
    # No blocking login wall — the app loads and is browsable by default
    # (preset/field/strategy setup). Actions that touch real data — uploading
    # report files, running extractions, comparisons, saving — each prompt to
    # sign in only when the user actually triggers them.
    init_state()
    ss = st.session_state
    username = ss.get("username")

    # Load persisted state for this user on first run of this session
    if username and not ss.get("_state_loaded"):
        _load_user_session(username, ss)
        ss._state_loaded = True

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 Pathology Extractor")
        sidebar_auth_widget()  # sign-in status; login form only if signed out

        if username:
            n_cached = 0 if ss.disk_cache is None else len(ss.disk_cache)
            with st.expander(f"🗑️ Cache · {n_cached} entr{'y' if n_cached == 1 else 'ies'}"):
                st.caption("Your cache is private to your account — clearing it just means "
                           "the next run of anything you've already extracted will call the "
                           "API again instead of returning instantly.")
                if st.button("Clear my cache", use_container_width=True, disabled=n_cached == 0):
                    cf = _cache_file(username)
                    if cf is not None and cf.exists():
                        cf.unlink()
                    ss.disk_cache = load_cache(username)
                    st.success(f"Cleared {n_cached} cached result{'s' if n_cached != 1 else ''}.")
                    st.rerun()

        st.divider()

        # ── API config ──────────────────────────────────────────────────────
        # ── Step 1: Provider ────────────────────────────────────────────────
        # Track the previously-selected provider explicitly in session_state
        # rather than reverse-looking it up from ss.api_url — Azure's URL is
        # user-editable free text, so once it's changed away from the
        # placeholder, a URL->provider lookup can never match "Azure OpenAI"
        # again, making the change-detection below fire on every single
        # rerun (which kept wiping the API key field).
        provider_names = list(PROVIDERS.keys())
        _cur_provider  = ss.get("_last_provider", "OpenAI")
        provider = st.selectbox("**Provider**", provider_names,
                                index=provider_names.index(_cur_provider),
                                help="Pick a provider — URL and model list update automatically.")
        cfg = PROVIDERS[provider]
        if provider != _cur_provider:
            ss.api_url   = cfg["url"]
            ss.api_model = cfg["models"][0] if cfg["models"][0] != "Custom…" else ""
            ss.api_key   = ""   # a key from the old provider won't work here — don't carry it over
            ss._api_key_draft = ""   # clear the unsaved draft too, same reason
            if provider == "Google Gemini" and ss.max_tokens < 2000:
                # Gemini's reasoning models spend a variable, often large
                # share of max_completion_tokens on internal "thinking"
                # before writing the answer — 600 (fine for non-reasoning
                # OpenAI models) reliably truncates the JSON mid-field.
                ss.max_tokens = 2000
        ss._last_provider = provider

        # ── Step 2: Model ────────────────────────────────────────────────────
        fetcher    = _MODEL_FETCHERS.get(provider)
        model_list = ss.live_models.get(provider) or cfg["models"]
        if "Custom…" not in model_list:
            model_list = model_list + ["Custom…"]
        # ss.api_model can end up holding a model id that belongs to a
        # *different* provider — e.g. a stale value restored from a saved
        # session whose provider/model got desynced (see the login-time
        # api_key race described above, which the same class of bug can hit).
        # Left alone, that falls into the "Custom…" box below and reads as an
        # intentional custom entry for this provider, when it's actually a
        # known model id that this provider's API will reject outright.
        # Reset it rather than let it masquerade as a valid custom value.
        _other_provider_models = {
            m for p, c in PROVIDERS.items() if p != provider
            for m in c["models"] if m != "Custom…"
        }
        if ss.api_model in _other_provider_models and ss.api_model not in model_list:
            ss.api_model = model_list[0] if model_list[0] != "Custom…" else ""

        if cfg["models"] == ["Custom…"] and provider not in ss.live_models:
            # Providers with no fixed catalog at all (currently just Azure
            # OpenAI) always need a typed-in value — routing through a
            # selectbox that offers exactly one option, "Custom…", just reads
            # as "no models available" rather than as a real choice. Skip
            # straight to the text input instead.
            st.markdown("**Deployment name**")
            ss.api_model = st.text_input(
                "Deployment name", ss.api_model,
                placeholder="e.g. gpt-4o-mini",
                label_visibility="collapsed",
                help="Azure has no fixed model list — deployment names are chosen by "
                     "whoever set up the resource, not by Azure or by this app. Ask your "
                     "Azure administrator, check the Azure Portal, or run "
                     "`az cognitiveservices account deployment list --name <resource> "
                     "--resource-group <group>` to see what's actually deployed.",
            )
        else:
            if fetcher is not None:
                lcol, rcol = st.columns([3, 1])
                with lcol:
                    st.markdown("**Model**")
                with rcol:
                    if st.button("🔄 Refresh", key="_refresh_models_btn", use_container_width=True,
                                 help=f"Fetch the current model list live from {provider}'s API "
                                      "instead of this app's built-in list, which can lag behind "
                                      "what the provider actually offers."):
                        fetch_key = ss.api_key or ss.get("_api_key_draft", "")
                        if provider == "OpenRouter (Claude / Gemini / Llama…)" or fetch_key:
                            try:
                                ss.live_models[provider] = fetcher(fetch_key)
                                ss.live_models_error.pop(provider, None)
                            except Exception as e:
                                ss.live_models_error[provider] = str(e)
                        else:
                            ss.live_models_error[provider] = "Add your API key below first, then refresh."
                if provider in ss.live_models:
                    st.caption(f"✅ Live list from {provider} · {len(ss.live_models[provider])} models")
                if provider in ss.live_models_error:
                    st.caption(f"⚠️ {ss.live_models_error[provider]}")
            else:
                st.markdown("**Model**")

            model_idx  = model_list.index(ss.api_model) \
                         if ss.api_model in model_list else model_list.index("Custom…")
            chosen_model = st.selectbox("Model", model_list, index=model_idx,
                                         label_visibility="collapsed")
            if chosen_model == "Custom…":
                ss.api_model = st.text_input("Custom model name", ss.api_model,
                                              placeholder="e.g. gemini-2.0-flash-exp",
                                              label_visibility="collapsed")
            else:
                ss.api_model = chosen_model

        # ── Step 3: API key ──────────────────────────────────────────────────
        # Shared key only works with OpenAI — force "My own key" for other providers
        shared_allowed = SHARED_ENABLED and provider == "OpenAI"
        if shared_allowed:
            ss.key_mode = st.radio("Key source", ["Shared key", "My own key"],
                                   index=0 if ss.key_mode == "Shared key" else 1,
                                   horizontal=True, label_visibility="collapsed")
        else:
            if ss.key_mode == "Shared key":
                ss.key_mode = "My own key"   # auto-switch when provider changes
            ss.key_mode = "My own key"

        if ss.key_mode == "Shared key":
            ss.api_key = SHARED_KEY
            ss.api_url = SHARED_URL
            used, left = ss.session_calls, SHARED_MAX - ss.session_calls
            st.progress(min(used / SHARED_MAX, 1.0),
                        text=f"{used} / {SHARED_MAX} calls used ({left} left)")
        else:
            if provider == "Azure OpenAI":
                st.caption("🔑 Paste a bearer token below — get one by running:")
                st.code(
                    "az account get-access-token "
                    "--resource https://cognitiveservices.azure.com "
                    "--query accessToken -o tsv",
                    language="bash",
                )
                st.caption(
                    "That command's entire output **is** the token — one long line starting "
                    "with `eyJ…`, no quotes or JSON around it. Copy that whole line and paste "
                    "it below. (If you instead ran the command without `--query accessToken "
                    "-o tsv` and got a `{ \"accessToken\": \"eyJ…\", \"expiresOn\": … }` block, "
                    "copy only the value between the quotes after `\"accessToken\":`.)"
                )
                st.caption(
                    "Requires `az login` first, and VPN if your Azure resource is on a "
                    "private network. The token is short-lived — if requests start failing "
                    "with an authentication error, just re-run the command and paste the new one."
                )
            else:
                st.caption(f"🔑 {cfg['key_help']}")
            # A form + explicit "Add" button, rather than binding ss.api_key
            # straight to the text_input: pasting/typing only stages a draft
            # here, so nothing is actually applied (and no request is sent)
            # until the user deliberately confirms it. Enter also submits,
            # since it's a form.
            with st.form(f"api_key_form_{provider}", border=False):
                kcol, bcol = st.columns([2, 1])
                with kcol:
                    key_draft = st.text_input("API key", key="_api_key_draft",
                                               type="password", placeholder=cfg["key_placeholder"],
                                               label_visibility="collapsed")
                with bcol:
                    add_clicked = st.form_submit_button("Add", use_container_width=True)
            if add_clicked:
                ss.api_key = key_draft
            st.caption("✅ Key is set — not saved anywhere; you'll need to re-enter it after signing out"
                       if ss.api_key else "⚠️ No key added yet")
            if provider == "Azure OpenAI":
                # Users were manually editing the full chat-completions URL
                # template (with a literal "<deployment>" placeholder) and
                # frequently leaving the placeholder in place, which made
                # every request 404/hang rather than fail obviously. Ask
                # only for the plain resource hostname instead, and build
                # the real path from the Deployment name already entered
                # in Step 2 — there's then nothing left to mistype.
                ss.azure_resource = st.text_input(
                    "Resource endpoint", ss.azure_resource,
                    placeholder="https://<your-resource>.openai.azure.com",
                    help="The base URL of your Azure OpenAI resource, from its "
                         "\"Keys and Endpoint\" page in the Azure Portal. No path — "
                         "the deployment path is built automatically from the "
                         "Deployment name above.",
                )
                resource   = ss.azure_resource.strip().rstrip("/")
                deployment = ss.api_model.strip()
                ss.api_url = (
                    f"{resource}/openai/deployments/{quote(deployment, safe='')}"
                    "/chat/completions?api-version=2024-02-01"
                    if resource and deployment else ""
                )
                if resource and deployment:
                    st.caption(f"Full endpoint: `{ss.api_url}`")
            else:
                st.caption(f"Endpoint: `{cfg['url']}`")

            # "Remember" is opt-in and off by default — the key/token is
            # encrypted at rest with a key derived from the account password
            # (see auth.py), but only signed-in users have that key at all.
            if username:
                _had_saved = load_user_secret(username) is not None
                remember = st.checkbox(
                    "💾 Remember this on this device (encrypted)",
                    value=_had_saved,
                    help="Encrypts your key/token with a key derived from your "
                         "account password and saves it to your private storage "
                         "folder, so you don't have to re-enter it next time you "
                         "sign in. Only you can decrypt it — resetting your "
                         "password makes any saved copy unrecoverable. For Azure, "
                         "the bearer token itself is still short-lived regardless "
                         "of whether it's saved — if requests start failing with "
                         "an authentication error, get a fresh token and re-save.",
                )
                if remember and ss.api_key:
                    save_user_secret(username, {
                        "provider": provider, "api_key": ss.api_key,
                        "api_url": ss.api_url, "api_model": ss.api_model,
                        "azure_resource": ss.azure_resource,
                    })
                elif not remember and _had_saved:
                    clear_user_secret(username)

        # ── Step 4: Temperature ──────────────────────────────────────────────
        _temp_supported = supports_temperature(ss.api_model)
        ss.temperature = st.slider(
            "Temperature", min_value=0.0, max_value=2.0,
            value=float(ss.temperature), step=0.1,
            disabled=not _temp_supported,
            help="0 = deterministic (recommended for extraction). "
                 "Higher values increase variability.",
        )
        if not _temp_supported:
            discovered = ss.api_model.lower().split("/")[-1] in _no_temp_discovered
            reason = "rejected temperature on first call" if discovered \
                     else "known reasoning model"
            st.caption(f"⚠️ `{ss.api_model}` does not support temperature "
                       f"({reason}) — omitted from request.")

        st.divider()

        # ── Reports file / Ground truth ─────────────────────────────────────
        # Uploading used to be allowed while signed out ("browsing is free"),
        # but that let anonymous visitors load and inspect real report data
        # with no account attached. Gate both uploaders behind a single
        # sign-in prompt, the same pattern used for running extractions.
        if require_login_for_action("upload files"):
            # Shown right at the point of action, not just once on page
            # load, and tailored to whichever provider is currently
            # selected — Azure (institutional network) vs. the three
            # public third-party APIs carry a materially different risk.
            if provider == "Azure OpenAI":
                st.caption(
                    "🔒 Azure OpenAI keeps report text within your "
                    "institution's own deployment — still confirm with your "
                    "IT/privacy office that this specific setup is approved "
                    "for PHI before uploading real patient reports."
                )
            else:
                st.warning(
                    f"⚠️ **{provider}** is a public third-party API — "
                    "uploaded report text is sent to it. Do not upload "
                    "reports with PHI. Upload only de-identified or "
                    "synthetic reports."
                )
            st.markdown("**📂 Reports file**")
            report_file = st.file_uploader("Excel file", type=["xlsx", "xls"],
                                            label_visibility="collapsed",
                                            key=f"report_uploader_{ss.uploader_epoch}")
            if report_file:
                fid = (report_file.name, report_file.size)
                if fid != ss._report_fid:
                    # Mark this file as "handled" whether or not it parses —
                    # otherwise a bad file keeps re-triggering the same crash
                    # on every rerun (any widget interaction anywhere on the
                    # page) for as long as it's still attached to the
                    # uploader, since fid never changes to let it pass.
                    ss._report_fid = fid
                    try:
                        df = pd.read_excel(report_file)
                    except Exception as e:
                        st.error(
                            f"Couldn't read **{report_file.name}** as an Excel "
                            f"file ({e}). Remove it above and upload a valid "
                            ".xlsx/.xls file — any previously loaded data is "
                            "unaffected."
                        )
                    else:
                        for col in df.select_dtypes(include="object").columns:
                            df[col] = df[col].apply(lambda v: clean_text(v) if isinstance(v, str) else v)
                        ss.report_df  = df
                        ss.results_df  = None
                        ss.review_idx  = 0
                        guessed_preset = guess_preset_from_filename(report_file.name)
                        if guessed_preset:
                            ss.preset_choice  = guessed_preset
                            ss.fields         = [dict(f) for f in PRESETS[guessed_preset]["fields"]]
                            ss.fields_version += 1
                            ss.gt_col_map     = {}  # reset mapping since fields just changed

            if ss.report_df is not None:
                cols = ss.report_df.columns.tolist()
                id_guess   = next((c for c in cols if c.lower() in ("id","case_id","caseid")), cols[0])
                text_guess = next((c for c in cols
                                   if any(k in c.lower() for k in ("report","text","note"))), cols[-1])
                ss.id_col   = st.selectbox("Case ID column", cols,
                                            index=cols.index(ss.id_col or id_guess))
                ss.text_col = st.selectbox("Report text column", cols,
                                            index=cols.index(ss.text_col or text_guess))
                n_cases = ss.report_df[ss.id_col].nunique()
                n_rows  = len(ss.report_df)
                st.caption(f"**{n_cases}** patients · {n_rows} sub-reports loaded")

                # PHI pre-pass: local regex scan, re-run only when the file
                # or the selected columns change (not on every rerun).
                scan_key = (ss._report_fid, ss.id_col, ss.text_col)
                if scan_key != ss._phi_scan_key:
                    ss._phi_scan_key  = scan_key
                    ss._phi_scan_hits = phi_scan.scan_dataframe(ss.report_df, ss.id_col, ss.text_col)
                n_hits = len(ss._phi_scan_hits) if ss._phi_scan_hits is not None else 0
                if n_hits:
                    n_flagged_cases = ss._phi_scan_hits["Case ID"].nunique()
                    st.warning(
                        f"🔎 PHI pre-pass flagged {n_hits} possible identifier(s) in "
                        f"{n_flagged_cases} case(s) — local pattern scan, not a "
                        "guarantee. Review before proceeding."
                    )
                    with st.expander("Show flagged snippets"):
                        st.dataframe(ss._phi_scan_hits, use_container_width=True, hide_index=True)
                else:
                    st.caption("✅ PHI pre-pass: no obvious identifier patterns detected (local scan).")

            st.divider()

            st.markdown("**📊 Ground truth** *(optional)*")
            gt_file = st.file_uploader("GT Excel file", type=["xlsx", "xls"],
                                        label_visibility="collapsed",
                                        key=f"gt_uploader_{ss.uploader_epoch}")
            if gt_file:
                fid = (gt_file.name, gt_file.size)
                if fid != ss._gt_fid:
                    ss._gt_fid = fid   # see Reports file above — set before parsing
                    try:
                        gt_df = pd.read_excel(gt_file)
                    except Exception as e:
                        st.error(
                            f"Couldn't read **{gt_file.name}** as an Excel file "
                            f"({e}). Remove it above and upload a valid "
                            ".xlsx/.xls file — any previously loaded data is "
                            "unaffected."
                        )
                    else:
                        ss.gt_df = gt_df
                        ss.gt_col_map = {}

            if ss.gt_df is not None:
                gt_cols    = ss.gt_df.columns.tolist()
                id_guess   = next((c for c in gt_cols if c.lower() in ("id","case_id","caseid")), gt_cols[0])
                ss.gt_id_col = st.selectbox("GT case ID column", gt_cols,
                                             index=gt_cols.index(ss.gt_id_col or id_guess))
                st.caption(f"{len(ss.gt_df)} rows · {len(gt_cols)} columns")

                if ss.fields:
                    # Force the initial open/closed state exactly once (open
                    # if nothing's mapped yet); afterward, leave it alone so
                    # mapping one field doesn't force-collapse it before the
                    # user has mapped the rest.
                    expander_kwargs = {}
                    if not ss._gt_mapping_seen:
                        expander_kwargs["expanded"] = not ss.gt_col_map
                        ss._gt_mapping_seen = True
                    with st.expander("🗂 GT column mapping", **expander_kwargs):
                        choices = ["(none)"] + gt_cols
                        for i, f in enumerate(ss.fields):
                            default = ss.gt_col_map.get(f["name"]) or f.get("gt_col", "")
                            idx = choices.index(default) if default in choices else 0
                            chosen = st.selectbox(
                                f["name"], choices, index=idx,
                                key=f"gtmap_{i}_{ss.fields_version}",
                                label_visibility="visible",
                            )
                            ss.gt_col_map[f["name"]] = chosen if chosen != "(none)" else ""

    # ── Guided navigation (segmented control + Back/Next) ──────────────────────
    # st.tabs can't be switched from Python, so the page is driven by
    # ss.nav_page instead. The segmented control uses a STABLE key ("nav_pill")
    # so Streamlit's own widget-state tracking handles direct pill clicks.
    # Back/Next (bottom) only ever mutate the plain ss.nav_page var and rerun
    # — they set _nav_sync_pill so that on the NEXT run, _render_nav_top()
    # pushes ss.nav_page into nav_pill BEFORE the widget is instantiated
    # (legal: that's before creation this run, not after). Without that
    # flag, nav_pill must be left alone here, since on a run where the user
    # clicked a pill directly, Streamlit has already written their click
    # into ss.nav_pill before the script starts — overwriting it here would
    # silently discard that click and revert to the old page.
    def _render_nav_top():
        if ss.pop("_nav_sync_pill", False):
            ss.nav_pill = NAV_PAGES[ss.nav_page]
        choice = st.segmented_control(
            "Navigate", NAV_PAGES, key="nav_pill",
            label_visibility="collapsed",
        )
        if choice is not None:
            ss.nav_page = NAV_PAGES.index(choice)

    def _render_nav_bottom():
        nc1, nc2, nc3 = st.columns([1, 8, 1])
        with nc1:
            if st.button("◀ Back", use_container_width=True,
                         disabled=ss.nav_page == 0, key=f"nav_back_{ss.nav_page}"):
                ss.nav_page = max(0, ss.nav_page - 1)
                ss._nav_sync_pill = True
                st.rerun()
        with nc3:
            if st.button("Next ▶", use_container_width=True,
                         disabled=ss.nav_page == len(NAV_PAGES) - 1, key=f"nav_next_{ss.nav_page}"):
                ss.nav_page = min(len(NAV_PAGES) - 1, ss.nav_page + 1)
                ss._nav_sync_pill = True
                st.rerun()

    _render_nav_top()
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 1 — Fields & Prompt
    # ══════════════════════════════════════════════════════════════════════════
    if ss.nav_page == 0:  # Fields & Prompt
        if ss.report_df is None:
            st.info("You can set up your preset and fields here first — "
                     "👈 upload a report file in the sidebar whenever you're ready to run extraction.")
            with st.expander("ℹ️ Quick start"):
                st.markdown("""
1. Pick a **Preset** below and hit **Apply** (or define custom fields).
2. Upload your **Excel file** (sidebar) — needs a Case ID column and a Report text column.
3. Go to **▶️ Run & Results** — hit **Run Extraction**.
4. Use **✏️ Review** to correct any errors and download the results.
                """)

        left, right = st.columns([4, 6], gap="large")

        with left:
            # Preset picker
            st.markdown("#### Preset")
            col_pre, col_btn = st.columns([3, 2])
            with col_pre:
                preset_name = st.selectbox("Preset", list(PRESETS.keys()),
                                           key="preset_choice",
                                           label_visibility="collapsed")
            with col_btn:
                if st.button("Apply", use_container_width=True, type="secondary"):
                    ss.fields         = [dict(f) for f in PRESETS[preset_name]["fields"]]
                    ss.fields_version += 1
                    ss.gt_col_map     = {}  # reset mapping when preset changes
                    st.rerun()

            # Field editor
            n_fields = len(ss.fields)
            st.markdown(f"#### Fields &nbsp; <span style='font-size:.8em;color:gray'>{n_fields} defined</span>",
                        unsafe_allow_html=True)
            if ss.fields:
                fields_df = pd.DataFrame(
                    [{"Name": f["name"], "Options": f["options"]} for f in ss.fields]
                )
                edited = st.data_editor(
                    fields_df, num_rows="dynamic", use_container_width=True,
                    key=f"fe_{ss.fields_version}",
                    column_config={
                        "Name":    st.column_config.TextColumn("Field name", width="medium"),
                        "Options": st.column_config.TextColumn("Options (/-separated)", width="large"),
                    },
                )
                new_fields = []
                for _, row in edited.iterrows():
                    nm = str(row.get("Name", "")).strip()
                    if not nm or nm == "nan":
                        continue
                    meta = next((f for f in ss.fields if f["name"] == nm), {})
                    new_fields.append({
                        "name": nm,
                        "options": str(row.get("Options", "")).strip(),
                        **{k: meta[k] for k in ("gt_col", "gt_norm") if k in meta},
                    })
                ss.fields = new_fields
            else:
                st.info("Apply a preset above, or add rows using the **＋** button.")

        with right:
            st.markdown("#### Strategy")
            strat_keys = list(STRATEGIES.keys())
            ss.strategy = st.radio(
                "Strategy", strat_keys,
                index=strat_keys.index(ss.strategy),
                format_func=lambda k: STRATEGIES[k],
                label_visibility="collapsed",
                horizontal=False,
            )
            st.caption(STRATEGY_HINTS.get(ss.strategy, ""))

            st.markdown("#### Prompt preview")
            if ss.fields:
                auto_prompt = build_prompt(ss.fields, ss.strategy)
                ss.prompt_override = st.checkbox("Override prompt manually", ss.prompt_override)
                if ss.prompt_override:
                    if not ss.custom_prompt:
                        ss.custom_prompt = auto_prompt
                    ss.custom_prompt = st.text_area(
                        "Custom prompt", ss.custom_prompt, height=340,
                        label_visibility="collapsed",
                    )
                    # Named saves are additive — they never touch the
                    # built-in PRESETS or overwrite each other's prompt text,
                    # just add/replace one named entry in the signed-in
                    # user's own saved_prompts dict.
                    if username:
                        with st.expander(f"💾 Saved prompts · {len(ss.saved_prompts)}"):
                            name_col, save_col = st.columns([2, 1])
                            with name_col:
                                new_name = st.text_input(
                                    "Save current prompt as", key="_save_prompt_name",
                                    placeholder="e.g. Stricter grading rules",
                                    label_visibility="collapsed",
                                )
                            with save_col:
                                if st.button("Save as…", use_container_width=True,
                                             disabled=not new_name.strip()):
                                    ss.saved_prompts[new_name.strip()] = ss.custom_prompt
                                    save_user_state(username, "saved_prompts", ss.saved_prompts)
                                    st.success(f"Saved “{new_name.strip()}”.")
                                    st.rerun()
                            if ss.saved_prompts:
                                st.divider()
                                chosen = st.selectbox(
                                    "Load a saved prompt", list(ss.saved_prompts.keys()),
                                    label_visibility="collapsed",
                                )
                                lcol, dcol = st.columns(2)
                                with lcol:
                                    if st.button("Load", use_container_width=True):
                                        ss.custom_prompt = ss.saved_prompts[chosen]
                                        st.rerun()
                                with dcol:
                                    if st.button("Delete", use_container_width=True):
                                        del ss.saved_prompts[chosen]
                                        save_user_state(username, "saved_prompts", ss.saved_prompts)
                                        st.rerun()
                            else:
                                st.caption("No saved prompts yet — name one above and click Save.")
                else:
                    st.text_area("Prompt (read-only)", auto_prompt,
                                 height=340, disabled=True,
                                 label_visibility="collapsed")
                    ss.custom_prompt = ""
                st.caption(f"{len(ss.custom_prompt or auto_prompt):,} characters")
            else:
                st.info("Define fields to preview the prompt.")

    # Resolve prompt and strategy for use in other tabs
    _prompt = ss.custom_prompt if ss.prompt_override and ss.custom_prompt \
              else (build_prompt(ss.fields, ss.strategy) if ss.fields else "")
    _strategy = ss.strategy

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 2 — Run & Results
    # ══════════════════════════════════════════════════════════════════════════
    if ss.nav_page == 1:  # Run & Results
        if ss.report_df is None:
            st.info("👈  Upload a pathology report Excel file in the sidebar to run extraction.")
        else:
            # ── Settings bar ──────────────────────────────────────────────────
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
                with c1:
                    sample_size = st.number_input("Sample size *(0 = all)*",
                                                   0, 50000, 0, step=10,
                                                   help="Limit to the first N unique cases. 0 runs all.")
                with c2:
                    ss.max_tokens = st.number_input("Max tokens", 200, 8000,
                                                     ss.max_tokens, step=100,
                                                     help="Reasoning models (e.g. Gemini) spend part of this "
                                                          "budget on internal thinking before the answer — "
                                                          "raise this if runs come back truncated.")
                with c3:
                    rate_limit = st.number_input("Pause (sec)", 0.0, 5.0, 0.5, step=0.25,
                                                 help="Seconds to wait between API calls.")
                with c4:
                    st.markdown(f"**Strategy:** {STRATEGIES.get(_strategy, _strategy)}")
                    st.caption(STRATEGY_HINTS.get(_strategy, ""))
                force_rerun = st.checkbox(
                    "🔁 Force re-run (ignore cache)", value=False,
                    help="Skip the cache and call the API for every case, even ones you've "
                         "already extracted — useful for checking how much the output varies "
                         "across repeated calls on the same report and prompt.",
                )

            key_ready = bool(ss.api_key) and (
                ss.key_mode == "Shared key" or          # shared key is always valid
                (ss.key_mode == "My own key" and         # own key must be non-empty
                 ss.api_key != SHARED_KEY)               # and not accidentally the shared key
            )
            can_run = bool(ss.fields) and key_ready and bool(ss.id_col)
            total_cases  = ss.report_df[ss.id_col].nunique()
            n_to_run     = min(sample_size, total_cases) if sample_size > 0 else total_cases
            run_col, dl_col = st.columns([3, 1])
            with run_col:
                run_btn = st.button(
                    f"▶  Run Extraction  —  {n_to_run} / {total_cases} case{'s' if total_cases != 1 else ''}",
                    type="primary", use_container_width=True, disabled=not can_run,
                )
            with dl_col:
                if ss.results_df is not None:
                    _dl_df = apply_corrections(ss.results_df, ss.corrections)
                    n_corr  = sum(1 for c in _dl_df["Case_ID"] if ss.corrections.get(c))
                    label   = f"⬇ CSV" + (f" ({n_corr} corrected)" if n_corr else "")
                    st.download_button(
                        label, _dl_df.to_csv(index=False).encode(),
                        file_name=f"extraction_{datetime.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv", use_container_width=True,
                    )

            if not can_run and not ss.results_df:
                if not ss.fields:
                    st.warning("Define fields in **📋 Fields & Prompt** first.")
                elif not key_ready:
                    st.warning(
                        f"Enter your own API key for **{ss.api_model}** "
                        f"(provider: {ss._last_provider}) "
                        "in the sidebar. The shared key only works with OpenAI."
                    )

            # ── Run ────────────────────────────────────────────────────────────
            # Each user's cache is private to them (see load_cache/append_cache),
            # so a signed-out visitor never has a cache of their own to hit —
            # running always requires signing in, cached or not.
            if run_btn and can_run and require_login_for_action("run extractions"):
                df = ss.report_df.copy()
                if sample_size > 0:
                    df = df[df[ss.id_col].isin(df[ss.id_col].unique()[:sample_size])]

                n_total, n_cases = len(df), df[ss.id_col].nunique()
                fnames    = [f["name"] for f in ss.fields]
                rows, id_cnts, seen = [], {}, set()
                new_calls, cases_done = 0, 0
                log_lines = []

                prog   = st.progress(0, text="Starting…")
                status = st.empty()

                for i, (_, row) in enumerate(df.iterrows()):
                    # iterrows (not itertuples) — itertuples renames any
                    # column whose name isn't a valid Python identifier (e.g.
                    # "Case ID", with a space) to a positional "_1", "_2", …,
                    # silently breaking getattr(row, ss.id_col) for any sheet
                    # whose headers aren't single bare words.
                    rid  = clean_text(str(row[ss.id_col]))
                    note = clean_text(str(row[ss.text_col]))
                    id_cnts[rid] = id_cnts.get(rid, 0) + 1
                    is_new = rid not in seen
                    if is_new:
                        seen.add(rid)
                        cases_done += 1

                    pct = (i + 1) / n_total
                    prog.progress(pct, text=f"Case {cases_done}/{n_cases}  ·  {rid}")
                    status.caption(f"Sub-report {id_cnts[rid]} for **{rid}**")

                    if ss.key_mode == "Shared key" and new_calls >= SHARED_MAX - ss.session_calls:
                        st.warning("Session quota reached — results saved for completed cases.")
                        break

                    extr = extract_one(note, _prompt, _strategy, ss.fields,
                                       ss.api_url, ss.api_key, ss.api_model,
                                       ss.max_tokens, ss.disk_cache, username,
                                       temperature=ss.temperature, use_cache=not force_rerun)
                    src = extr.get("source", "")
                    if src != "disk":
                        new_calls += extr.get("n_calls", 1)
                        ss.disk_cache = load_cache(username)
                    if extr.get("error"):
                        log_lines.append(f"❌ {rid}: {extr['error']}")
                    else:
                        log_lines.append(f"{'💾' if src=='disk' else '🌐'} {rid}")

                    rows.append({"Case_ID": rid, **parse_json(extr.get("json"), fnames)})
                    if rate_limit > 0 and src != "disk":
                        time.sleep(rate_limit)

                ss.session_calls += new_calls
                prog.progress(1.0, text="Done!")
                status.empty()
                ss.results_df = collapse_subreports(rows, fnames)
                ss.review_idx  = 0
                if username:   # signed-out cache-only runs have no account to persist to
                    _save_user_session(username, ss)   # persist results + settings
                st.success(
                    f"✅  Extracted **{len(ss.results_df)} cases**"
                    f"  ·  {new_calls} new API call{'s' if new_calls != 1 else ''}"
                    f"  ·  {len(rows) - new_calls} from cache"
                )
                with st.expander("Run log"):
                    st.text("\n".join(log_lines))

            # ── Results ────────────────────────────────────────────────────────
            if ss.results_df is not None:
                _flags_df = consistency_check.check_dataframe(ss.fields, ss.results_df)
                if len(_flags_df):
                    n_flagged_cases = _flags_df["Case_ID"].nunique()
                    st.warning(
                        f"⚠️ Consistency check found {len(_flags_df)} flag(s) in "
                        f"{n_flagged_cases} case(s) — local check on the extracted "
                        "fields above, not a new LLM call. Review before trusting "
                        "the flagged values."
                    )
                    with st.expander("Show consistency flags"):
                        st.dataframe(_flags_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("✅ Consistency check: no out-of-vocabulary values or "
                               "lymph-node contradictions found (local check).")

                sub_table, sub_acc = st.tabs(["📋 Per-case results", "📊 Accuracy"])

                with sub_acc:
                    if ss.gt_df is not None and any(ss.gt_col_map.values()):
                        acc_df = compute_accuracy(ss.results_df, ss.gt_df, ss.fields,
                                                  ss.gt_id_col, ss.gt_col_map)
                        if acc_df is not None:
                            global_acc = round(
                                (acc_df["Accuracy (%)"] * acc_df["N"]).sum() / acc_df["N"].sum(), 1)
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Global accuracy", f"{global_acc}%")
                            m2.metric("Cases evaluated", len(ss.results_df))
                            m3.metric("Fields evaluated", len(acc_df))

                            with st.expander("🎯 Classification metrics (sensitivity / specificity / PPV / NPV)"):
                                st.caption(
                                    "Pick a positive class for any field below to add its "
                                    "sensitivity, specificity, PPV, and NPV to the table. "
                                    "Leave a field unset to skip it — these metrics only "
                                    "apply to fields with a clear positive/negative meaning."
                                )
                                eval_fields = [f for f in ss.fields if f["name"] in set(acc_df["Field"])]
                                for f in eval_fields:
                                    fname = f["name"]
                                    opts = [o.strip() for o in f.get("options", "").split("/") if o.strip()]
                                    is_numeric = any(o.lower() in ("integer", "number", "numeric") for o in opts)
                                    if is_numeric:
                                        checked = st.checkbox(
                                            f"{fname}: value > 0 = positive",
                                            value=(ss.positive_class.get(fname) == ">0"),
                                            key=f"posclass_{fname}",
                                        )
                                        if checked:
                                            ss.positive_class[fname] = ">0"
                                        else:
                                            ss.positive_class.pop(fname, None)
                                    else:
                                        cat_opts = [o for o in opts if o.lower() != "not mentioned"]
                                        current = ss.positive_class.get(fname, [])
                                        current = [v for v in current if v in cat_opts]
                                        sel = st.multiselect(
                                            f"{fname}: positive value(s)",
                                            cat_opts, default=current, key=f"posclass_{fname}",
                                        )
                                        if sel:
                                            ss.positive_class[fname] = sel
                                        else:
                                            ss.positive_class.pop(fname, None)

                            cls_metrics = compute_classification_metrics(
                                ss.results_df, ss.gt_df, ss.fields,
                                ss.gt_id_col, ss.gt_col_map, ss.positive_class,
                            )
                            if cls_metrics:
                                for col in ("Sensitivity (%)", "Specificity (%)", "PPV (%)", "NPV (%)"):
                                    acc_df[col] = acc_df["Field"].map(
                                        lambda fn: cls_metrics.get(fn, {}).get(col))

                            # Color-coded per-field bar
                            fig = px.bar(
                                acc_df.sort_values("Accuracy (%)"),
                                x="Accuracy (%)", y="Field", orientation="h",
                                color="Accuracy (%)",
                                color_continuous_scale="RdYlGn",
                                range_color=[0, 100],
                                text="Accuracy (%)",
                            )
                            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                            fig.update_layout(
                                showlegend=False, coloraxis_showscale=False,
                                margin=dict(l=0, r=60, t=10, b=10),
                                height=max(250, 35 * len(acc_df)),
                                yaxis_title="", xaxis_range=[0, 110],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(acc_df.sort_values("Accuracy (%)", ascending=False),
                                         use_container_width=True, hide_index=True)
                        else:
                            st.info("No GT columns matched. Check the **GT column mapping** in the sidebar.")
                    else:
                        st.info("Upload a **ground truth file** in the sidebar and set the column mapping to see accuracy.")

                with sub_table:
                    _corrected_df = apply_corrections(ss.results_df, ss.corrections)
                    n_corr = sum(1 for c in _corrected_df["Case_ID"] if ss.corrections.get(c))
                    st.dataframe(_corrected_df, use_container_width=True, hide_index=True,
                                 height=min(600, 38 + 35 * len(_corrected_df)))
                    if n_corr:
                        st.caption(f"✅ {n_corr} case(s) show corrected values from the Review tab.")
                    st.download_button(
                        f"⬇ Download results (CSV){f'  ·  {n_corr} corrected' if n_corr else ''}",
                        _corrected_df.to_csv(index=False).encode(),
                        file_name=f"extraction_{datetime.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 3 — Compare
    # ══════════════════════════════════════════════════════════════════════════
    if ss.nav_page == 2:  # Compare Strategies
        sub_strat, sub_models = st.tabs(["📈 Strategies", "🤖 Models"])

        with sub_strat:
            cc1, cc2 = st.columns([1, 3], gap="large")

            with cc1:
                st.markdown("#### Strategies")
                st.caption("General")
                for k in ("cot", "cov", "zero_shot"):
                    checked = k in ss.compare_selected
                    if st.checkbox(STRATEGIES[k], value=checked, key=f"cmp_{k}"):
                        if k not in ss.compare_selected:
                            ss.compare_selected.append(k)
                    else:
                        if k in ss.compare_selected:
                            ss.compare_selected.remove(k)

                st.caption("Persona-consistent")
                for k in ("persona_cot", "persona_cov", "persona_fewshot",
                          "persona_strict", "persona_zeroshot"):
                    checked = k in ss.compare_selected
                    if st.checkbox(STRATEGIES[k], value=checked, key=f"cmp_{k}"):
                        if k not in ss.compare_selected:
                            ss.compare_selected.append(k)
                    else:
                        if k in ss.compare_selected:
                            ss.compare_selected.remove(k)

                st.divider()
                cmp_sample = st.number_input("Cases (0 = all)", 0, 50000, 10, step=5,
                                             help="Number of unique cases per strategy.")
                cmp_rate   = st.number_input("Pause (sec)", 0.0, 5.0, 0.5, step=0.25,
                                             key="cmp_rate")
                n_sel = len(ss.compare_selected)
                cmp_btn = st.button(
                    f"▶  Compare  {n_sel} strateg{'ies' if n_sel != 1 else 'y'}",
                    type="primary", use_container_width=True, disabled=n_sel < 2,
                )
                if n_sel < 2:
                    st.caption("Select ≥ 2 strategies.")

            with cc2:
                if cmp_btn and require_login_for_action("compare strategies"):
                    if not ss.api_key:
                        st.error("Enter an API key.")
                    elif not ss.fields:
                        st.error("Define fields in Fields & Prompt tab.")
                    elif ss.report_df is None:
                        st.error("Upload a report file in the sidebar first.")
                    else:
                        df = ss.report_df.copy()
                        if cmp_sample > 0:
                            df = df[df[ss.id_col].isin(df[ss.id_col].unique()[:cmp_sample])]

                        fnames   = [f["name"] for f in ss.fields]
                        all_rows, cmp_log = [], []
                        n_cases   = df[ss.id_col].nunique()
                        total     = len(ss.compare_selected) * n_cases
                        done      = 0
                        new_calls = 0
                        quota_hit = False
                        prog2     = st.progress(0, text="Starting…")

                        for strat in ss.compare_selected:
                            if quota_hit:
                                break
                            prompt = build_prompt(ss.fields, strat)
                            rows, id_cnts, seen = [], {}, set()

                            for _, row in df.iterrows():
                                # Same shared-key budget Run Extraction enforces
                                # (ss.session_calls) — without this check, Compare
                                # Strategies could burn through the shared key's
                                # real API budget with no cap at all.
                                if ss.key_mode == "Shared key" and \
                                   new_calls >= SHARED_MAX - ss.session_calls:
                                    st.warning("Session quota reached — "
                                               "results saved for completed cases.")
                                    quota_hit = True
                                    break

                                rid  = clean_text(str(row[ss.id_col]))
                                note = clean_text(str(row[ss.text_col]))
                                id_cnts[rid] = id_cnts.get(rid, 0) + 1
                                if rid not in seen:
                                    seen.add(rid)
                                    done += 1
                                    prog2.progress(done / total,
                                                   text=f"{STRATEGIES[strat]}  ·  {rid}")

                                extr = extract_one(note, prompt, strat, ss.fields,
                                                   ss.api_url, ss.api_key, ss.api_model,
                                                   ss.max_tokens, ss.disk_cache, username,
                                                   temperature=ss.temperature)
                                if extr.get("source") != "disk":
                                    new_calls += extr.get("n_calls", 1)
                                    ss.disk_cache = load_cache(username)
                                    if cmp_rate > 0:
                                        time.sleep(cmp_rate)
                                if extr.get("error"):
                                    cmp_log.append(f"❌ {strat}/{rid}: {extr['error']}")
                                rows.append({"Case_ID": rid,
                                             **parse_json(extr.get("json"), fnames)})

                            collapsed = collapse_subreports(rows, fnames)
                            collapsed["strategy"] = strat
                            all_rows.append(collapsed)

                        ss.session_calls += new_calls
                        prog2.progress(1.0, text="Done!")
                        ss.compare_df = pd.concat(all_rows, ignore_index=True)
                        st.success(f"Comparison complete — {len(ss.compare_selected)} strategies.")
                        if cmp_log:
                            with st.expander(f"⚠️ {len(cmp_log)} errors"):
                                st.text("\n".join(cmp_log))

                if ss.compare_df is not None:
                    has_gt = ss.gt_df is not None and any(ss.gt_col_map.values())
                    if has_gt:
                        strats_in = ss.compare_df["strategy"].unique().tolist()
                        fnames    = [f["name"] for f in ss.fields]
                        heat = {}
                        for s in strats_in:
                            sub = ss.compare_df[ss.compare_df["strategy"] == s].drop(columns="strategy")
                            a   = compute_accuracy(sub, ss.gt_df, ss.fields,
                                                   ss.gt_id_col, ss.gt_col_map)
                            if a is not None:
                                heat[STRATEGIES.get(s, s)] = dict(zip(a["Field"], a["Accuracy (%)"]))

                        if heat:
                            heat_df = pd.DataFrame(heat, index=fnames).T
                            fig = px.imshow(
                                heat_df, text_auto=True,
                                color_continuous_scale="RdYlGn", range_color=[0, 100],
                                aspect="auto",
                                labels={"x": "Field", "y": "Strategy", "color": "Acc %"},
                            )
                            fig.update_traces(texttemplate="%{z:.0f}%")
                            fig.update_layout(
                                xaxis_tickangle=-35,
                                margin=dict(l=0, r=0, t=30, b=10),
                                height=max(300, 70 * len(heat)),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Global accuracy per strategy
                            summary = []
                            for s_label, field_acc in heat.items():
                                vals = [v for v in field_acc.values() if v is not None]
                                summary.append({"Strategy": s_label,
                                                "Mean accuracy (%)": round(sum(vals)/len(vals), 1) if vals else None})
                            st.dataframe(pd.DataFrame(summary).sort_values("Mean accuracy (%)", ascending=False),
                                         use_container_width=True, hide_index=True)
                    else:
                        st.info("Upload a ground truth file to see accuracy. Raw extractions below.")
                        st.dataframe(ss.compare_df, use_container_width=True, hide_index=True)

                    st.download_button(
                        "⬇ Download comparison (CSV)",
                        ss.compare_df.to_csv(index=False).encode(),
                        file_name=f"comparison_{datetime.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )

        # ── Model comparison sub-tab ───────────────────────────────────────────────
        with sub_models:
            if ss.report_df is None or not ss.fields:
                st.info("Upload a report file and define fields first.")
            else:
                mc1, mc2 = st.columns([1, 3], gap="large")

                with mc1:
                    st.markdown("#### Strategy (fixed)")
                    strat_keys = list(STRATEGIES.keys())
                    ss.model_compare_strategy = st.selectbox(
                        "Strategy", strat_keys,
                        index=strat_keys.index(ss.model_compare_strategy),
                        format_func=lambda k: STRATEGIES[k],
                        label_visibility="collapsed",
                    )
                    st.divider()
                    st.markdown("#### Models to compare")
                    st.caption("Pick ≥ 2 models. One API key covers every model checked under it.")

                    catalog_by_provider = {}
                    for entry in MODEL_CATALOG:
                        catalog_by_provider.setdefault(entry["provider"], []).append(entry)

                    for prov, entries in catalog_by_provider.items():
                        st.caption(prov)
                        for entry in entries:
                            mid = entry["id"]
                            checked = mid in ss.model_compare_selected
                            if st.checkbox(entry["model"], value=checked, key=f"mcmp_{mid}"):
                                if mid not in ss.model_compare_selected:
                                    ss.model_compare_selected.append(mid)
                            else:
                                if mid in ss.model_compare_selected:
                                    ss.model_compare_selected.remove(mid)

                    selected_entries    = [e for e in MODEL_CATALOG if e["id"] in ss.model_compare_selected]
                    selected_providers  = sorted({e["provider"] for e in selected_entries})

                    if selected_providers:
                        st.divider()
                        st.markdown("###### API keys")
                        for prov in selected_providers:
                            ss.model_compare_keys[prov] = st.text_input(
                                f"{prov} key", ss.model_compare_keys.get(prov, ""),
                                type="password", key=f"mcmpkey_{prov}",
                            )

                    st.divider()
                    cmp_m_sample = st.number_input("Cases (0 = all)", 0, 50000, 10, step=5,
                                                   key="cmp_m_sample")
                    cmp_m_rate   = st.number_input("Pause (sec)", 0.0, 5.0, 0.5, step=0.25,
                                                   key="cmp_m_rate")

                    valid_rows = [e for e in selected_entries
                                  if ss.model_compare_keys.get(e["provider"])]
                    run_mcmp = st.button(
                        f"▶  Compare {len(valid_rows)} model{'s' if len(valid_rows)!=1 else ''}",
                        type="primary", use_container_width=True,
                        disabled=len(valid_rows) < 2,
                    )
                    if len(valid_rows) < 2:
                        st.caption("Check ≥ 2 models and enter an API key for each provider used.")

                with mc2:
                    if run_mcmp and require_login_for_action("compare models"):
                        if not ss.fields:
                            st.error("Define fields first.")
                        else:
                            df = ss.report_df.copy()
                            if cmp_m_sample > 0:
                                df = df[df[ss.id_col].isin(df[ss.id_col].unique()[:cmp_m_sample])]

                            fnames   = [f["name"] for f in ss.fields]
                            prompt   = build_prompt(ss.fields, ss.model_compare_strategy)
                            all_rows, m_log = [], []
                            total    = len(valid_rows) * df[ss.id_col].nunique()
                            done     = 0
                            prog     = st.progress(0, text="Starting…")

                            for entry in valid_rows:
                                # Model id alone (e.g. "openai/gpt-5") is
                                # normally unambiguous without the provider
                                # prefix, and is much more readable as a
                                # heatmap row label.
                                nickname = entry["model"]
                                model    = entry["model"]
                                api_url  = entry["url"]
                                api_key  = ss.model_compare_keys[entry["provider"]]
                                rows, id_cnts, seen = [], {}, set()

                                for _, row in df.iterrows():
                                    rid  = clean_text(str(row[ss.id_col]))
                                    note = clean_text(str(row[ss.text_col]))
                                    id_cnts[rid] = id_cnts.get(rid, 0) + 1
                                    if rid not in seen:
                                        seen.add(rid)
                                        done += 1
                                        prog.progress(done / total,
                                                      text=f"{nickname}  ·  {rid}")

                                    extr = extract_one(note, prompt,
                                                       ss.model_compare_strategy,
                                                       ss.fields, api_url, api_key, model,
                                                       ss.max_tokens, ss.disk_cache, username,
                                                       temperature=ss.temperature)
                                    if extr.get("error"):
                                        m_log.append(f"❌ {nickname}/{rid}: {extr['error']}")
                                    if extr.get("source") != "disk":
                                        ss.disk_cache = load_cache(username)
                                        if cmp_m_rate > 0:
                                            time.sleep(cmp_m_rate)
                                    rows.append({"Case_ID": rid,
                                                 **parse_json(extr.get("json"), fnames)})

                                collapsed = collapse_subreports(rows, fnames)
                                collapsed["model"] = nickname
                                all_rows.append(collapsed)

                            prog.progress(1.0, text="Done!")
                            ss.model_compare_df = pd.concat(all_rows, ignore_index=True)
                            st.success(f"Model comparison complete — {len(valid_rows)} models.")
                            if m_log:
                                with st.expander(f"⚠️ {len(m_log)} errors"):
                                    st.text("\n".join(m_log))

                    if ss.model_compare_df is not None:
                        has_gt = ss.gt_df is not None and any(ss.gt_col_map.values())
                        if has_gt:
                            models_in = ss.model_compare_df["model"].unique().tolist()
                            fnames    = [f["name"] for f in ss.fields]
                            heat = {}
                            for m in models_in:
                                sub = ss.model_compare_df[
                                    ss.model_compare_df["model"] == m].drop(columns="model")
                                a = compute_accuracy(sub, ss.gt_df, ss.fields,
                                                     ss.gt_id_col, ss.gt_col_map)
                                if a is not None:
                                    heat[m] = dict(zip(a["Field"], a["Accuracy (%)"]))

                            if heat:
                                heat_df = pd.DataFrame(heat, index=fnames).T
                                fig = px.imshow(
                                    heat_df, text_auto=True,
                                    color_continuous_scale="RdYlGn", range_color=[0, 100],
                                    aspect="auto",
                                    labels={"x": "Field", "y": "Model", "color": "Acc %"},
                                )
                                fig.update_traces(texttemplate="%{z:.0f}%")
                                fig.update_layout(
                                    xaxis_tickangle=-35,
                                    margin=dict(l=0, r=0, t=30, b=10),
                                    height=max(300, 80 * len(heat)),
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                summary = [{"Model": m,
                                            "Mean accuracy (%)": round(
                                                sum(v for v in acc.values() if v is not None) /
                                                max(len([v for v in acc.values() if v is not None]), 1), 1)}
                                           for m, acc in heat.items()]
                                st.dataframe(pd.DataFrame(summary).sort_values(
                                    "Mean accuracy (%)", ascending=False),
                                    use_container_width=True, hide_index=True)
                        else:
                            st.info("Upload a ground truth file to see accuracy. Raw table below.")

                        st.dataframe(ss.model_compare_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "⬇ Download model comparison (CSV)",
                            ss.model_compare_df.to_csv(index=False).encode(),
                            file_name=f"model_comparison_{datetime.today().strftime('%Y%m%d')}.csv",
                            mime="text/csv", use_container_width=True,
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 4 — Review
    # ══════════════════════════════════════════════════════════════════════════
    if ss.nav_page == 3:  # Review
        if ss.results_df is None:
            st.info("Run extraction first (**▶️ Run & Results** tab).")
        else:

            case_ids   = ss.results_df["Case_ID"].tolist()
            n_cases    = len(case_ids)
            n_corrected = sum(1 for c in case_ids if ss.corrections.get(c))

            # ── Navigation bar ─────────────────────────────────────────────────
            nav1, nav2, nav3, nav4 = st.columns([1, 4, 1, 2])
            with nav1:
                if st.button("◀ Prev", use_container_width=True,
                             disabled=ss.review_idx == 0):
                    ss.review_idx -= 1
                    st.rerun()
            with nav2:
                sel = st.selectbox(
                    "Case", case_ids,
                    index=min(ss.review_idx, n_cases - 1),
                    label_visibility="collapsed",
                )
                ss.review_idx = case_ids.index(sel)
            with nav3:
                if st.button("Next ▶", use_container_width=True,
                             disabled=ss.review_idx >= n_cases - 1):
                    ss.review_idx += 1
                    st.rerun()
            with nav4:
                corrected_flag = "✅" if ss.corrections.get(sel) else "○"
                st.markdown(
                    f"**{corrected_flag}  Case {ss.review_idx+1} / {n_cases}**"
                    f"  &nbsp;·&nbsp;  {n_corrected} corrected",
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Main content: report | fields ──────────────────────────────────
            r_col, f_col = st.columns([5, 7], gap="large")

            with r_col:
                rh1, rh2 = st.columns([4, 1])
                with rh1:
                    st.markdown("**Original report**")
                if ss.report_df is not None and ss.id_col and ss.text_col:
                    matches = ss.report_df[ss.report_df[ss.id_col].astype(str) == sel]
                    parts   = [clean_text(str(t)) for t in matches[ss.text_col]]
                    report_text = "\n\n─── sub-report ───\n\n".join(parts) if parts else "(not found)"
                else:
                    report_text = "(report file not loaded)"
                # If a field's "🧠 Explain this case" quote was located in
                # report_text (see evidence_reasoning.find_quote_span), that
                # span is highlighted by default (search, if used, takes over
                # instead -- see _render_report_pane).
                active_span = None
                if ss.review_highlight and ss.review_highlight[0] == sel:
                    hl_exp = ss.case_explanations.get(sel, {}).get(ss.review_highlight[1])
                    if hl_exp:
                        active_span = hl_exp.get("span")

                @st.dialog("Original report", width="large")
                def _expanded_report_dialog():
                    st.caption(f"Case {sel}")
                    _render_report_pane(report_text, active_span,
                                         key_prefix=f"{sel}_expand", height="72vh")

                with rh2:
                    if st.button("⛶ Expand", key=f"expand_report_{sel}",
                                  use_container_width=True):
                        _expanded_report_dialog()

                _render_report_pane(report_text, active_span, key_prefix=f"{sel}_main")

            with f_col:
                st.markdown("**Extracted fields — type in Corrected to override**")
                case_row = ss.results_df[ss.results_df["Case_ID"] == sel]
                fnames   = ([f["name"] for f in ss.fields] if ss.fields
                            else [c for c in ss.results_df.columns if c != "Case_ID"])
                corr     = ss.corrections.get(sel, {})

                # ── AI reasoning: one on-demand call for the whole case ────────
                field_values = {}
                for fname in fnames:
                    if not case_row.empty and fname in case_row.columns:
                        v = case_row[fname].iloc[0]
                        if v is not None and str(v).strip() and \
                           str(v).lower() not in ("nan", "none", "not mentioned"):
                            field_values[fname] = str(v)

                exp_key_ready = bool(ss.api_key) and (
                    ss.key_mode == "Shared key" or
                    (ss.key_mode == "My own key" and ss.api_key != SHARED_KEY))
                already_explained = sel in ss.case_explanations
                exp_col1, exp_col2 = st.columns([2, 5])
                with exp_col1:
                    explain_clicked = st.button(
                        "🧠 Re-explain this case" if already_explained else "🧠 Explain this case",
                        disabled=not exp_key_ready or not field_values,
                        use_container_width=True,
                    )
                with exp_col2:
                    st.caption("One call asks the model what in the report supports each "
                               "value. Its own explanation, generated after the fact — a "
                               "starting point for review, not proof. Click 📍 next to a "
                               "field to highlight its supporting text in the report.")
                if explain_clicked:
                    if ss.key_mode == "Shared key" and 1 > SHARED_MAX - ss.session_calls:
                        st.error("Shared-key session quota reached — switch to your own key.")
                    else:
                        with st.spinner("Asking the model to explain…"):
                            result = evidence_reasoning.explain_case(
                                report_text, field_values, call_api,
                                ss.api_url, ss.api_key, ss.api_model)
                        if result["error"] is None:
                            ss.session_calls += 1
                            ss.case_explanations[sel] = result["explanations"]
                            ss.review_highlight = None  # spans may have shifted/changed
                        else:
                            st.error(f"Explanation failed: {result['error']}")

                # Header row
                h1, h2, h3 = st.columns([3, 3, 3])
                h1.markdown("**Field**")
                h2.markdown("**Extracted**")
                h3.markdown("**Corrected**")
                st.divider()

                # One text_input per field — updates session_state on every keystroke,
                # so the Save button always reads the latest value (no timing issues).
                for fname in fnames:
                    extracted = ""
                    if not case_row.empty and fname in case_row.columns:
                        v = case_row[fname].iloc[0]
                        extracted = "" if (v is None or str(v).lower() in ("nan","none","")) else str(v)
                    c1, c2, c3 = st.columns([3, 3, 3])
                    c1.markdown(f"<small>{fname}</small>", unsafe_allow_html=True)
                    c2.markdown(f"<small>{extracted or '—'}</small>", unsafe_allow_html=True)
                    with c3:
                        st.text_input(
                            fname, value=corr.get(fname, ""),
                            key=f"corr_{sel}_{fname}",
                            placeholder="leave blank to keep extracted",
                            label_visibility="collapsed",
                        )
                    explanation = ss.case_explanations.get(sel, {}).get(fname)
                    if explanation:
                        exp_text    = explanation["text"]
                        unsupported = exp_text.strip().upper().startswith("UNSUPPORTED")
                        cap_col, loc_col = st.columns([9, 1])
                        with cap_col:
                            st.caption(f"{'⚠️' if unsupported else '🧠'} {exp_text}")
                        with loc_col:
                            if explanation.get("span"):
                                is_active = ss.review_highlight == (sel, fname)
                                if st.button(
                                    "📍" if not is_active else "✖️",
                                    key=f"locate_{sel}_{fname}",
                                    help="Un-highlight" if is_active
                                         else "Highlight the supporting text in the report",
                                ):
                                    ss.review_highlight = None if is_active else (sel, fname)
                                    st.rerun()

                st.divider()
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("💾 Save corrections", type="primary", use_container_width=True) \
                            and require_login_for_action("save corrections"):
                        saved = {}
                        for fname in fnames:
                            val = str(st.session_state.get(f"corr_{sel}_{fname}", "")).strip()
                            if val and val.lower() not in ("nan", "none", ""):
                                saved[fname] = val
                        ss.corrections[sel] = saved
                        _save_user_session(username, ss)   # persist corrections
                        ss._save_msg = (
                            f"✅ {len(saved)} correction(s) saved for {sel}"
                            + (f" at {datetime.now().strftime('%H:%M:%S')}" if saved else "")
                        )
                        st.rerun()
                with b2:
                    if st.button("🗑 Clear this case", use_container_width=True):
                        ss.corrections.pop(sel, None)
                        # Reset text_input session state for this case
                        for fname in fnames:
                            key = f"corr_{sel}_{fname}"
                            if key in st.session_state:
                                del st.session_state[key]
                        ss._save_msg = f"🗑 Corrections cleared for {sel}"
                        st.rerun()

                if ss._save_msg:
                    st.info(ss._save_msg)

            # Always show download — corrections applied if any exist
            st.divider()
            _final_df = apply_corrections(ss.results_df, ss.corrections)
            # Strip markdown bold markers from save message for use in button label
            _save_hint = re.sub(r"\*\*", "", ss._save_msg) if ss._save_msg else ""
            _label = (f"⬇  Download results CSV  ·  {_save_hint}"
                      if _save_hint else "⬇  Download results CSV  (no corrections yet)")
            st.download_button(
                _label, _final_df.to_csv(index=False).encode(),
                file_name=f"results_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )
    # ══════════════════════════════════════════════════════════════════════════
    # Tab 5 — Cohort
    # ══════════════════════════════════════════════════════════════════════════
    if ss.nav_page == 4:  # Cohort
        if ss.results_df is None:
            st.info("Run extraction first (**▶️ Run & Results** tab).")
        else:

            fnames    = [f["name"] for f in ss.fields] if ss.fields \
                        else [c for c in ss.results_df.columns if c != "Case_ID"]
            work_df   = apply_corrections(ss.results_df, ss.corrections)

            # ── Section 1: Data quality heatmap ───────────────────────────────
            st.markdown("### 🗂 Data quality — extracted values overview")
            st.caption("Each cell shows the extracted value. "
                       "🟢 green = extracted · 🟡 amber = not mentioned · 🔴 red = missing")

            z, cell_text, hover_text, case_ids = build_data_quality_heatmap(work_df, fnames)

            # Discrete colorscale: 0=red, 1=amber, 2=green
            colorscale = [
                [0.000, "#d62728"], [0.333, "#d62728"],
                [0.334, "#ff7f0e"], [0.666, "#ff7f0e"],
                [0.667, "#2ca02c"], [1.000, "#2ca02c"],
            ]
            cell_h = max(28, min(48, 400 // max(len(work_df), 1)))
            fig_miss = go.Figure(go.Heatmap(
                z=z, x=fnames, y=case_ids,
                text=cell_text, texttemplate="%{text}",
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>",
                colorscale=colorscale, zmin=0, zmax=2,
                showscale=False,
                xgap=2, ygap=2,
            ))
            fig_miss.update_layout(
                xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=11)),
                margin=dict(l=10, r=10, t=10, b=10),
                height=max(280, cell_h * len(work_df) + 80),
            )
            st.plotly_chart(fig_miss, use_container_width=True)

            # Legend
            leg1, leg2, leg3 = st.columns(3)
            leg1.markdown("🟢 **Extracted** — value found in report")
            leg2.markdown("🟡 **Not mentioned** — field absent from report")
            leg3.markdown("🔴 **Missing** — extraction failed or not run")

            # Summary counts
            flat_z       = [v for row in z for v in row]
            total_cells  = len(flat_z)
            extracted    = flat_z.count(2)
            not_ment     = flat_z.count(1)
            missing      = flat_z.count(0)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Extracted",      f"{extracted} ({100*extracted//total_cells}%)")
            mc2.metric("Not mentioned",  f"{not_ment}  ({100*not_ment//total_cells}%)")
            mc3.metric("Missing",        f"{missing}   ({100*missing//total_cells}%)")

            st.divider()

            # ── Section 2: Case / Control definition ──────────────────────────
            st.markdown("### 👥 Case / Control definition")
            def field_value_choices(field_name):
                """Sorted unique non-null values from the results for a given field."""
                if not field_name or field_name not in work_df.columns:
                    return []
                vals = (work_df[field_name]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .unique()
                        .tolist())
                return sorted([v for v in vals if v.lower() not in ("nan", "none", "")])

            def value_widget(field_name, op, current_val, key):
                """Show a selectbox of known values, or a text input for list/no-value ops."""
                no_value_ops = ("is missing", "is present")
                list_ops     = ("in (comma list)", "not in (comma list)")
                if op in no_value_ops:
                    st.caption("*(no value needed)*")
                    return ""
                if op in list_ops:
                    return st.text_input("Value", current_val,
                                         placeholder="e.g. R0, R1",
                                         label_visibility="collapsed", key=key)
                choices = field_value_choices(field_name)
                if choices:
                    options = choices + ["Custom…"]
                    # Determine which option to pre-select
                    if current_val in choices:
                        idx = options.index(current_val)
                    elif current_val:
                        idx = options.index("Custom…")
                    else:
                        idx = 0
                    chosen = st.selectbox("Value", options, index=idx,
                                          label_visibility="collapsed", key=key + "_sel")
                    if chosen == "Custom…":
                        return st.text_input("Custom value", current_val,
                                             label_visibility="collapsed", key=key + "_txt")
                    return chosen
                # Fallback: no data yet → plain text input
                return st.text_input("Value", current_val,
                                      placeholder="type a value",
                                      label_visibility="collapsed", key=key)

            field_choices = ["(none)"] + fnames
            cc1, cc2 = st.columns(2, gap="large")

            with cc1:
                st.markdown("**Case condition**")
                cf1, cf2, cf3 = st.columns([3, 2, 3])
                case_field = cf1.selectbox("Field##case", field_choices,
                    index=field_choices.index(ss.case_cond["field"])
                          if ss.case_cond["field"] in field_choices else 0,
                    label_visibility="collapsed")
                case_op = cf2.selectbox("Op##case", OPERATORS,
                    index=OPERATORS.index(ss.case_cond["op"])
                          if ss.case_cond["op"] in OPERATORS else 0,
                    label_visibility="collapsed")
                with cf3:
                    case_val = value_widget(
                        case_field if case_field != "(none)" else "",
                        case_op, ss.case_cond["value"], "case_val")
                ss.case_cond = {"field": case_field if case_field != "(none)" else "",
                                "op": case_op, "value": case_val}

            with cc2:
                st.markdown("**Control condition**")
                cf1, cf2, cf3 = st.columns([3, 2, 3])
                ctrl_field = cf1.selectbox("Field##ctrl", field_choices,
                    index=field_choices.index(ss.ctrl_cond["field"])
                          if ss.ctrl_cond["field"] in field_choices else 0,
                    label_visibility="collapsed")
                ctrl_op = cf2.selectbox("Op##ctrl", OPERATORS,
                    index=OPERATORS.index(ss.ctrl_cond["op"])
                          if ss.ctrl_cond["op"] in OPERATORS else 0,
                    label_visibility="collapsed")
                with cf3:
                    ctrl_val = value_widget(
                        ctrl_field if ctrl_field != "(none)" else "",
                        ctrl_op, ss.ctrl_cond["value"], "ctrl_val")
                ss.ctrl_cond = {"field": ctrl_field if ctrl_field != "(none)" else "",
                                "op": ctrl_op, "value": ctrl_val}

            st.divider()

            # ── Section 3: Inclusion / Exclusion rules ─────────────────────────
            st.markdown("### 🔍 Inclusion / Exclusion criteria")
            st.caption("Each rule is applied in order. Use **Exclude if** to remove cases "
                       "that match, or **Include only if** to drop cases that don't match.")

            with st.expander("🗣️ Describe your cohort in plain English (optional)"):
                st.caption(
                    "Describe criteria in your own words — e.g. \"liver cases with "
                    "vascular invasion and R1 margins\" — and this proposes rules for "
                    "the table below from your extracted fields. Always reviewed before "
                    "anything is added; it never changes the table on its own."
                )
                nl_query = st.text_input(
                    "Cohort description", key="nl_cohort_query",
                    placeholder="e.g. liver cases with vascular invasion and R1 margins",
                    label_visibility="collapsed")

                nl_key_ready = bool(ss.api_key) and (
                    ss.key_mode == "Shared key" or
                    (ss.key_mode == "My own key" and ss.api_key != SHARED_KEY))
                already_translated = bool(
                    ss.nl_cohort_proposal
                    and ss.nl_cohort_proposal.get("query") == nl_query.strip()
                    and not ss.nl_cohort_proposal.get("error")
                )
                translate_clicked = st.button(
                    "✨ Translate", disabled=not nl_key_ready or not nl_query.strip()
                                             or already_translated)
                if not nl_key_ready:
                    st.caption("Enter an API key in the sidebar to use this.")

                if translate_clicked:
                    if ss.key_mode == "Shared key" and 1 > SHARED_MAX - ss.session_calls:
                        ss.nl_cohort_proposal = {
                            "query": nl_query.strip(), "rules": [], "n_dropped": 0,
                            "error": "Shared-key session quota reached.",
                        }
                    else:
                        with st.spinner("Translating…"):
                            result = nl_cohort.translate_query(
                                nl_query.strip(), ss.fields, OPERATORS,
                                call_api, ss.api_url, ss.api_key, ss.api_model)
                        if result["error"] is None:
                            ss.session_calls += 1
                        ss.nl_cohort_proposal = {"query": nl_query.strip(), **result}

                if ss.nl_cohort_proposal:
                    prop = ss.nl_cohort_proposal
                    if prop.get("error"):
                        st.error(f"Translation failed: {prop['error']}")
                    elif not prop["rules"]:
                        st.info("No rules could be confidently mapped to your extracted "
                                "fields. Try rephrasing, or build rules manually below.")
                    else:
                        st.markdown(f"**Proposed rules for:** _{prop['query']}_")
                        st.dataframe(pd.DataFrame(prop["rules"]),
                                     use_container_width=True, hide_index=True)
                        if prop["n_dropped"]:
                            st.caption(f"⚠️ {prop['n_dropped']} phrase(s) couldn't be "
                                       "mapped to a valid field/operator and were dropped.")
                        if st.button("➕ Add these rules to the table below"):
                            ss.ie_rules = ss.ie_rules + prop["rules"]
                            ss.ie_rules_version += 1
                            ss.nl_cohort_proposal = None
                            st.rerun()

            ie_default = pd.DataFrame(ss.ie_rules if ss.ie_rules else
                [{"Field": "", "Operator": "==", "Value": "", "Action": "Exclude if"}])
            ie_edited = st.data_editor(
                ie_default,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Field":    st.column_config.SelectboxColumn(
                                    "Field", options=fnames, width="medium"),
                    "Operator": st.column_config.SelectboxColumn(
                                    "Operator", options=OPERATORS, width="medium"),
                    "Value":    st.column_config.TextColumn("Value", width="medium"),
                    "Action":   st.column_config.SelectboxColumn(
                                    "Action",
                                    options=["Exclude if", "Include only if"],
                                    width="medium"),
                },
                key=f"ie_rules_editor_{ss.ie_rules_version}",
            )
            ss.ie_rules = ie_edited.dropna(subset=["Field"]).to_dict("records")

            st.divider()

            # ── Apply & results ────────────────────────────────────────────────
            if st.button("▶  Build cohort", type="primary", use_container_width=True):
                ss.cohort_df = build_cohort(ss.results_df, ss.corrections,
                                            ss.case_cond, ss.ctrl_cond, ss.ie_rules)
                st.rerun()

            if ss.cohort_df is not None:
                cdf = ss.cohort_df
                n_total   = len(cdf)
                n_case    = int((cdf["Group"] == "Case").sum())
                n_ctrl    = int((cdf["Group"] == "Control").sum())
                n_uncls   = int((cdf["Group"] == "Unclassified").sum())
                n_incl    = int((cdf["Cohort"] == "Included").sum())
                n_excl    = int((cdf["Cohort"] == "Excluded").sum())

                st.markdown("#### Results")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Total",        n_total)
                r2.metric("Cases",        n_case,  delta=f"{100*n_case//max(n_total,1)}%",
                          delta_color="off")
                r3.metric("Controls",     n_ctrl,  delta=f"{100*n_ctrl//max(n_total,1)}%",
                          delta_color="off")
                r4.metric("Included",     n_incl)
                r5.metric("Excluded",     n_excl)

                # Exclusion breakdown
                if n_excl > 0:
                    excl_counts = (cdf[cdf["Cohort"]=="Excluded"]["Exclusion reason"]
                                   .value_counts().reset_index())
                    excl_counts.columns = ["Reason", "N"]
                    with st.expander(f"Exclusion breakdown ({n_excl} cases)"):
                        st.dataframe(excl_counts, use_container_width=True, hide_index=True)

                # Group distribution chart
                grp_counts = cdf["Group"].value_counts().reset_index()
                grp_counts.columns = ["Group", "N"]
                color_map  = {"Case": "#d62728", "Control": "#1f77b4",
                              "Unclassified": "#7f7f7f"}
                fig_grp = px.bar(grp_counts, x="Group", y="N", color="Group",
                                 color_discrete_map=color_map, text="N",
                                 height=280)
                fig_grp.update_layout(showlegend=False, margin=dict(t=20, b=10))
                st.plotly_chart(fig_grp, use_container_width=True)

                # Full table
                with st.expander("View cohort table"):
                    st.dataframe(cdf, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇  Download cohort CSV",
                    cdf.to_csv(index=False).encode(),
                    file_name=f"cohort_{datetime.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True,
                )


    # ══════════════════════════════════════════════════════════════════════════
    # Tab 5 — Help
    # ══════════════════════════════════════════════════════════════════════════
    if ss.nav_page == 5:  # Help
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
### Quick start
1. **Upload** your Excel file in the sidebar (needs a Case ID column and a Report text column).
2. **Fields & Prompt** — apply the PRAD or Tongue Cancer preset, or define custom fields.
3. **Run & Results** — hit ▶ Run Extraction. Accuracy appears automatically if GT is loaded.
4. **Compare Strategies** — select ≥ 2 strategies to see a side-by-side heatmap.
5. **Review** — navigate case-by-case, edit the *Corrected* column, and download.
6. **Cohort** — build Inclusion/Exclusion rules and export the matching subset.

### Ground truth evaluation
Upload a GT Excel file (sidebar). Set the **GT column mapping** expander to map each extracted field to the corresponding GT column. For PRAD this is pre-filled from the preset.

### Caching
Results are cached per account (each signed-in user has their own private cache — never shared with other users). Re-running the same report text and prompt returns your own cached result instantly at no cost. Signed-out browsing has no cache, since there's no account to keep it private to.

### PHI pre-pass
After you upload a reports file, a local regex scan checks for common identifier patterns — SSNs, phone numbers, emails, labeled MRN/DOB/name fields, street addresses — and flags any hits right in the sidebar before you run extraction. It's a heuristic safety net, not certified de-identification: a clean scan means "no obvious red flags," not a PHI guarantee. Upload only de-identified or synthetic reports regardless of what the scan shows.

### Consistency check
After extraction, Run & Results checks the extracted fields for internal contradictions: values outside a field's own defined options (e.g. a stray `pMX` where only `pM0/pM1a/pM1b/pM1c` are valid), more positive lymph nodes than nodes examined, or N-Stage `N0` paired with a nonzero positive-node count. It's a local check on already-extracted values, not another LLM call, and it never modifies anything — just flags cases worth a closer look in Review.

### Explain this case
In the Review tab, **🧠 Explain this case** asks the model — one call for the whole case — what in the report supports each extracted value, and to say plainly when it's a direct quote versus an inference, unit conversion, or interpretation. A field flagged ⚠️ **UNSUPPORTED** means the model itself doesn't think the report backs up that value — worth a close look. This is the model's own after-the-fact explanation, not proof: treat it as a starting point for review. When the model's supporting quote can be found verbatim (or near-verbatim) in the report, a **📍** button appears next to that field — click it to highlight the matching span in the report pane on the left, and jump to it if it's out of view. No 📍 button means no matching span was found (often the case for UNSUPPORTED fields, or values reached by inference rather than a quotable phrase).

### Searching and expanding the report
Above the report pane in the Review tab, the **🔍 find in report** box highlights every match in the text and steps between them with **◀ ▶**; searching this way takes over from the 📍 highlight until the search box is cleared. **⛶ Expand** opens the same report — with its own search box — in a larger pop-up window, useful for long reports where the side-by-side pane feels cramped.

### Natural-language cohort builder
In the Cohort tab, the **"Describe your cohort in plain English"** expander turns a sentence like *"liver cases with vascular invasion and R1 margins"* into Inclusion/Exclusion rules — one LLM call, translated using your actual extracted fields and their allowed values. It always shows a preview table first; nothing is added to the rules below until you click **Add these rules**.
            """)
        with c2:
            st.markdown("""
### Prompting strategies

| Strategy | Description |
|---|---|
| Chain-of-Thought | Step-by-step reasoning before JSON |
| Chain-of-Verif. | Two calls: extract → verify |
| Persona · CoT | Persona + reasoning inside JSON |
| Persona · CoV | Persona + two-pass verification |
| Persona · Few-shot | Persona + one worked example |
| Persona · Strict | Persona + 5 strict rules |
| Persona · Zero-shot | Expert pathologist persona |
| Zero-shot | Paper-exact prompt, no persona |

### API compatibility
Works with any OpenAI-format endpoint:
- **OpenAI** — default URL, any `gpt-*` model
- **Azure OpenAI** — Entra ID bearer token, Deployment name, and Resource endpoint (see the full guide below)
- **Google Gemini**, **OpenRouter** — paste a provider API key
            """)

        st.divider()
        st.markdown("### Citing PathEx")
        st.markdown("If PathEx contributes to your research, please cite the manuscript describing it:")
        st.code(
            "Zhuang Y, Alami V, Feser WJ, Gao D. PathEx: A Web-Based Multi-Strategy Platform for\n"
            "Large Language Model–Driven Structured Extraction from Pathology Reports.\n"
            "Manuscript in preparation.",
            language=None,
        )

        guide_html = _load_user_guide_html()
        if guide_html:
            st.divider()
            # The guide's own layout (sticky sidebar TOC, two-column panels)
            # is designed for a full browser window, not a small nested
            # iframe — hence the "open in a new tab" button. Getting a real
            # URL for that turned out to have two dead ends: a data: URL
            # doesn't work as a *top-level* navigation target (Chrome blocks
            # that as an anti-phishing measure — the same content is fine as
            # an iframe src, which is why the embed below works), and
            # Streamlit's own static-file server (app/static/, requires
            # enableStaticServing) hard-codes non-image/PDF extensions to
            # Content-Type: text/plain for XSS-prevention, so a served
            # .html file won't render either. A blob: URL sidesteps both —
            # it carries its own explicit type metadata rather than relying
            # on a server header, and isn't subject to the data: top-level
            # navigation block — hence building one client-side below.
            # The guide's own trailing <script> (scroll-spy TOC highlighting)
            # means its raw text contains a literal "</script" — embedded
            # verbatim inside *this* wrapping <script> tag, the HTML parser
            # (which scans for that literal sequence to end a script block
            # before the JS parser ever runs) would close this tag early and
            # break everything after it. Escaping the slash keeps the JS
            # string's runtime value identical while hiding it from the
            # HTML-parsing pass.
            guide_json = json.dumps(guide_html).replace("</script", "<\\/script")
            open_btn_html = f"""
              <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
                <button id="open-guide-btn" style="width:100%; padding:0.65rem; font-size:1rem;
                  font-weight:600; background:#ff4b4b; color:white; border:none; border-radius:8px;
                  cursor:pointer;">
                  📘 Open full user guide in a new tab
                </button>
              </div>
              <script>
                document.getElementById('open-guide-btn').addEventListener('click', function() {{
                  const blob = new Blob([{guide_json}], {{type: 'text/html;charset=utf-8'}});
                  window.open(URL.createObjectURL(blob), '_blank');
                }});
              </script>
            """
            components.html(open_btn_html, height=55)
            st.caption("If the button above doesn't open a new tab (some browsers restrict this), "
                       "use the guide embedded below instead.")
            components.html(guide_html, height=2200, scrolling=True)

    st.divider()
    _render_nav_bottom()


if __name__ == "__main__":
    main()
