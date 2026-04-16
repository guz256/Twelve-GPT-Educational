import os
import streamlit as st


def _secret_bool(key: str, default: bool = False) -> bool:
    value = st.secrets.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _normalize_azure_base(url: str) -> str:
    if not url:
        return url
    out = url.rstrip("/")
    # The legacy openai.ChatCompletion Azure flow expects only the resource root.
    for suffix in ("/openai/v1", "/openai"):
        if out.lower().endswith(suffix):
            out = out[: -len(suffix)]
            break
    return out


GPT_EMBEDDINGS_ENGINE = st.secrets.get("GPT_EMBEDDINGS_ENGINE") or st.secrets.get(
    "GPT_EMBEDDINGS_MODEL"
)
GPT_EMBEDDINGS_KEY = st.secrets.get("GPT_EMBEDDINGS_KEY")

GPT_DEFAULT = "3.5"
GPT3_BASE = st.secrets.get("GPT_BASE")
GPT3_VERSION = st.secrets.get("GPT_VERSION")
GPT3_KEY = st.secrets.get("GPT_KEY")
GPT3_ENGINE = st.secrets.get("GPT_ENGINE") or st.secrets.get("GPT_CHAT_MODEL")
GPT4_BASE = st.secrets.get('GPT4o_BASE')
GPT4_VERSION = st.secrets.get('GPT4o_VERSION')
GPT4_KEY = st.secrets.get('GPT4o_KEY')
GPT4_ENGINE = st.secrets.get("GPT4o_ENGINE") or st.secrets.get("GPT4o_CHAT_MODEL")

# Gemini secrets
USE_GEMINI = _secret_bool("USE_GEMINI", False)
GEMINI_API_KEY = (
    st.secrets.get("GEMINI_API_KEY", "")
    or os.getenv("GEMINI_API_KEY", "")
    or os.getenv("GOOGLE_API_KEY", "")
)
GEMINI_CHAT_MODEL = st.secrets.get("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = st.secrets.get(
    "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
)

if GPT_DEFAULT == "4":
    GPT_BASE = _normalize_azure_base(GPT4_BASE)
    GPT_VERSION = GPT4_VERSION
    GPT_KEY = GPT4_KEY
    GPT_ENGINE = GPT4_ENGINE
elif GPT_DEFAULT == "3.5":
    GPT_BASE = _normalize_azure_base(GPT3_BASE)
    GPT_VERSION = GPT3_VERSION
    GPT_KEY = GPT3_KEY
    GPT_ENGINE = GPT3_ENGINE
else:
    raise ValueError("GPT_DEFAULT must be '3.5' or '4'")
