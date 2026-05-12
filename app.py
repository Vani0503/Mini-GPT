"""
Streamlit: side-by-side Mini-GPT (Day 7 word-level model) vs OpenAI Chat.

Checkpoint: mini_gpt_checkpoint.pth with model_state_dict, stoi, itos, vocab_size
(as produced by your training script).

Secrets: set OPENAI_API_KEY in Streamlit Cloud secrets or env.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import streamlit as st
import torch
from openai import OpenAI

from mini_gpt_model import generate_words, load_mini_gpt_checkpoint


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_checkpoint_resource(checkpoint_path: str):
    device = get_device()
    model, meta = load_mini_gpt_checkpoint(checkpoint_path, device)
    return model, meta, device


def ensure_checkpoint(path_str: str, download_url: str | None) -> str | None:
    p = Path(path_str).expanduser()
    if p.is_file():
        return str(p.resolve())
    if download_url:
        cache_dir = Path(tempfile.gettempdir()) / "mini_gpt_streamlit"
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = Path(download_url.split("?")[0]).name or "mini_gpt_checkpoint.pth"
        dest = cache_dir / name
        if not dest.is_file():
            urlretrieve(download_url, dest)
        return str(dest)
    return None


def run_openai_chat(api_key: str, model_name: str, user_text: str, system_prompt: str) -> str:
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
    )
    return (r.choices[0].message.content or "").strip()


def init_session_state() -> None:
    if "turns" not in st.session_state:
        st.session_state.turns = []


st.set_page_config(page_title="Mini-GPT vs ChatGPT", layout="wide", page_icon="⚖️")
init_session_state()

st.title("Mini-GPT vs ChatGPT")
st.caption(
    "Mini-GPT uses your Day 7 checkpoint (word-level vocabulary). "
    "History stays until you clear it or close the session."
)

with st.sidebar:
    st.subheader("OpenAI")
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = (secret_key or env_key or st.text_input("OpenAI API key", type="password")).strip()
    openai_model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    system_prompt = st.text_area(
        "System prompt (ChatGPT)",
        value="You are a helpful assistant. Answer clearly and concisely.",
        height=100,
    )

    st.subheader("Mini-GPT checkpoint")
    default_ckpt = os.getenv("MINI_GPT_CHECKPOINT", "mini_gpt_checkpoint.pth")
    ckpt_path = st.text_input("Checkpoint path", value=default_ckpt)
    ckpt_url = st.text_input(
        "Optional: raw download URL",
        value=os.getenv("MINI_GPT_CHECKPOINT_URL", ""),
        help="If the .pth is not in the repo, provide a direct raw URL (small files; GitHub LFS often fails).",
    )

    st.subheader("Generation (matches training script)")
    max_new = st.slider("Max new tokens", 8, 256, 50, 8)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    top_k = st.slider("top_k (0 = sample from full vocab)", 0, 200, 20)

    if st.button("Clear chat history", type="secondary"):
        st.session_state.turns = []
        st.rerun()

resolved = ensure_checkpoint(ckpt_path, ckpt_url.strip() or None)
model_loaded = False
meta: dict = {}
model = None
device = get_device()

if resolved:
    try:
        model, meta, device = load_checkpoint_resource(resolved)
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        model_loaded = False
else:
    st.warning(
        f"Checkpoint not found at `{ckpt_path}`. "
        "Commit `mini_gpt_checkpoint.pth` next to `app.py` or set MINI_GPT_CHECKPOINT_URL."
    )

if model_loaded and model is not None:
    with st.sidebar:
        vs = meta.get("vocab_size", "?")
        bs = meta.get("block_size", "?")
        ed = meta.get("embed_dim", "?")
        st.success(f"Loaded on {device} | vocab={vs} | block={bs} | embed={ed}")

for i, turn in enumerate(st.session_state.turns):
    with st.container():
        st.markdown(f"**You** ({i + 1})")
        st.info(turn["user"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Mini-GPT")
            st.markdown(turn["mini"])
        with c2:
            st.markdown("### ChatGPT")
            st.markdown(turn["openai"])
        st.divider()

if not st.session_state.turns:
    st.markdown(
        "Messages are split on **whitespace** (same as training). "
        "Words not in your training vocabulary are skipped for Mini-GPT."
    )

user_input = st.chat_input("Message…")

if user_input and not api_key:
    st.warning("Add your OpenAI API key in the sidebar or set OPENAI_API_KEY in secrets.")

top_k_arg = int(top_k) if top_k > 0 else 0

if user_input and api_key and model_loaded and model is not None:
    mini_out = ""
    oa_out = ""
    with st.spinner("Running Mini-GPT…"):
        try:
            mini_out = generate_words(
                model,
                meta["stoi"],
                meta["itos"],
                user_input,
                device,
                block_size=int(meta["block_size"]),
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=top_k_arg,
            )
        except Exception as e:
            mini_out = f"[Mini-GPT error] {e}"
    with st.spinner("Calling ChatGPT…"):
        try:
            oa_out = run_openai_chat(api_key, openai_model, user_input, system_prompt)
        except Exception as e:
            oa_out = f"[OpenAI error] {e}"

    st.session_state.turns.append({"user": user_input, "mini": mini_out, "openai": oa_out})
    st.rerun()
