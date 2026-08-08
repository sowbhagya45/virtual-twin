"""
app.py — virtual-twin Streamlit Application
=============================================
Entry point for both local dev and Streamlit Community Cloud.

Local:
    streamlit run app.py

Streamlit Cloud:
    Push to GitHub → connect at share.streamlit.io → add secrets → deploy.
"""
from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# ── Environment ───────────────────────────────────────────────────────────────
# Locally: reads from .env
# Streamlit Cloud: reads from st.secrets (no .env needed)
load_dotenv()

def _load_secrets():
    """Bridge Streamlit secrets → environment variables.

    On Streamlit Cloud: reads from st.secrets (secrets.toml / Secrets panel).
    Locally without secrets.toml: silently skips — .env loaded above covers it.
    """
    try:
        for key in ("GOOGLE_API_KEY", "SENDGRID_API_KEY", "OWNER_EMAIL", "LANGSMITH_API_KEY", "GEMINI_MODEL"):
            if key not in os.environ and key in st.secrets:
                os.environ[key] = st.secrets[key]
    except Exception:
        # No secrets.toml present locally — environment is already loaded from .env
        pass

_load_secrets()


def _is_quota_error(exc: Exception) -> tuple[bool, int]:
    """Return (is_quota_error, retry_after_seconds) from a 429 exception."""
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        import re
        m = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
        wait = int(m.group(1)) if m else 60
        return True, wait
    return False, 0

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sowbhagya's Virtual Twin",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS: minimal clean look ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Header badge */
    .twin-badge {
        background: #1e3a5f;
        color: #e6edf3;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 8px;
    }
    /* Agent indicator strip */
    .agent-tag {
        font-size: 11px;
        color: #57606a;
        font-style: italic;
        margin-top: -6px;
        margin-bottom: 4px;
        padding-left: 4px;
    }
    /* Subtle divider */
    hr.twin-divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Graph bootstrap (cached per process) ─────────────────────────────────────
# Defined here (before the sidebar) so the model selector can call load_graph.clear().
@st.cache_resource(show_spinner="Loading virtual-twin agents...")
def load_graph():
    """Loads and compiles the LangGraph once per process."""
    # Run ingestion if chroma_db does not exist yet (first boot on Streamlit Cloud)
    if not os.path.exists("./chroma_db"):
        st.toast("Building knowledge base for the first time — this takes ~2 min...", icon="⚙️")
        from ingest import ingest
        ingest()
    from agents.graph import get_graph
    return get_graph()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="twin-badge">🤖 virtual-twin</div>', unsafe_allow_html=True)
    st.markdown("### Sowbhagya Mohanthy")
    st.markdown(
        "AI Engineer & Software Developer  \n"
        "IBM · Hyderabad, India  \n"
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sowbhagya-mohanthy-8a8a68221) "
        "[![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/Sowbhagya-45)"
    )
    st.markdown('<hr class="twin-divider">', unsafe_allow_html=True)

    # ── Model selector — TOP of sidebar so it's always visible ───────────────
    # 16 chat models confirmed on this API key (live-probed).
    # Each is a separate quota bucket — switch when one is exhausted.
    # Ordered: newest → oldest (try top ones first, they have separate buckets).
    _model_options = {
        # ── Gemini 3.x (fresh buckets, not yet used) ──────────────────────
        "gemini-3.5-flash       [3.x ✦ NEW]":       "models/gemini-3.5-flash",
        "gemini-3.5-flash-lite  [3.x ✦ NEW]":       "models/gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite  [3.x ✦ NEW]":       "models/gemini-3.1-flash-lite",
        "gemini-3-flash-preview [3.x preview]":      "models/gemini-3-flash-preview",
        "gemini-3-pro-preview   [3.x preview]":      "models/gemini-3-pro-preview",
        "gemini-3.1-pro-preview [3.x preview]":      "models/gemini-3.1-pro-preview",
        "gemini-3.6-flash       [3.x preview]":      "models/gemini-3.6-flash",
        # ── Gemini 2.5 (20 req/day — use sparingly) ───────────────────────
        "gemini-2.5-flash       [2.5 — 20/day]":    "models/gemini-2.5-flash",
        # ── Gemini 2.0 (1500 req/day, likely drained today) ───────────────
        "gemini-2.0-flash       [2.0]":              "models/gemini-2.0-flash",
        "gemini-2.0-flash-lite  [2.0]":              "models/gemini-2.0-flash-lite",
        "gemini-2.0-flash-001   [2.0 pinned]":       "models/gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001 [2.0 pinned]":    "models/gemini-2.0-flash-lite-001",
    }
    _current_model_env = os.environ.get("GEMINI_MODEL", "models/gemini-3.5-flash")
    _current_label = next(
        (k for k, v in _model_options.items() if v == _current_model_env),
        "gemini-3.5-flash       [3.x ✦ NEW]",
    )

    # Show a red banner when the last call hit quota
    if st.session_state.get("_quota_hit"):
        st.error("⚠️ Quota exhausted — switch model below", icon="🔴")

    _selected_label = st.selectbox(
        "🤖 Model (switch on 429)",
        list(_model_options.keys()),
        index=list(_model_options.keys()).index(_current_label),
        help=(
            "Each model is a separate free-tier bucket (~1500 req/day).\n"
            "When you see a 429 error, pick the next model in this list."
        ),
    )
    _selected_model = _model_options[_selected_label]
    if _selected_model != _current_model_env:
        os.environ["GEMINI_MODEL"] = _selected_model
        st.session_state.pop("_quota_hit", None)   # clear the red banner
        load_graph.clear()   # rebuild graph with new model on next invoke
        st.rerun()

    st.markdown('<hr class="twin-divider">', unsafe_allow_html=True)

    st.markdown("**What can I help with?**")
    st.markdown(
        "- 💼 Skills, experience & projects\n"
        "- 📚 Education & certifications\n"
        "- 📅 Book a meeting\n"
        "- 💬 General questions"
    )
    st.markdown('<hr class="twin-divider">', unsafe_allow_html=True)

    # Download resume button
    resume_path = os.path.join("data", "resume.pdf")
    if os.path.exists(resume_path):
        with open(resume_path, "rb") as _f:
            st.download_button(
                label="📄 Download Resume",
                data=_f,
                file_name="Sowbhagya_Mohanthy_Resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.markdown('<hr class="twin-divider">', unsafe_allow_html=True)

    # Show active agent name for transparency
    if "last_agent" in st.session_state:
        agent_labels = {
            "rag_agent": "📚 Profile agent",
            "scheduler_agent": "📅 Scheduler agent",
            "notifier_agent": "📬 Notifier agent",
            "chitchat_agent": "💬 Chit-chat agent",
            "none": "—",
        }
        label = agent_labels.get(st.session_state.last_agent, st.session_state.last_agent)
        st.caption(f"Last active agent: **{label}**")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        for key in ("messages", "thread_id", "last_agent", "knowledge_gap"):
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown('<hr class="twin-divider">', unsafe_allow_html=True)
    st.caption("Powered by LangGraph + Gemini")

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("## Hi, I'm Sowbhagya's virtual twin 👋")
st.markdown(
    "Ask me anything about Sowbhagya's background, skills, projects, or experience. "
    "I can also help you **book a call** with him or leave him a message."
)

# Quick-action suggestion chips (simulated with small buttons in columns)
_chip_cols = st.columns(4)
_chips = [
    ("💼 Key skills", "What are Sowbhagya's key skills?"),
    ("🚀 Top project", "Tell me about his flagship project"),
    ("📅 Book a call", "I'd like to schedule a meeting"),
    ("📬 Contact him", "I want to reach Sowbhagya directly"),
]
for col, (label, prompt_text) in zip(_chip_cols, _chips):
    with col:
        if st.button(label, use_container_width=True, key=f"chip_{label}"):
            st.session_state["_chip_prompt"] = prompt_text

st.markdown('<hr class="twin-divider">', unsafe_allow_html=True)

# ── Session state bootstrap ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "last_agent" not in st.session_state:
    st.session_state.last_agent = "none"


# ── Streaming helper ──────────────────────────────────────────────────────────
import time as _time

def _stream_text(text: str, delay: float = 0.015):
    """Yield words with a tiny delay to create a typewriter streaming effect."""
    for word in text.split(" "):
        yield word + " "
        _time.sleep(delay)


# ── Render existing conversation history ─────────────────────────────────────
_agent_icon = {
    "rag_agent": "🤖",
    "scheduler_agent": "📅",
    "notifier_agent": "📬",
    "chitchat_agent": "💬",
}

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        icon = _agent_icon.get(msg.get("agent", "rag_agent"), "🤖")
        with st.chat_message("assistant", avatar=icon):
            st.markdown(msg["content"])
            if msg.get("agent") and msg["agent"] != "rag_agent":
                agent_labels = {
                    "scheduler_agent": "📅 Scheduler agent",
                    "notifier_agent": "📬 Notifier agent",
                    "chitchat_agent": "💬 Chit-chat agent",
                }
                label = agent_labels.get(msg["agent"], msg["agent"])
                st.markdown(
                    f'<div class="agent-tag">handled by {label}</div>',
                    unsafe_allow_html=True,
                )

# ── Chat input ───────────────────────────────────────────────────────────────
# MUST be called unconditionally every render so Streamlit always shows the box.
# Never put inside a walrus-operator short-circuit — that hides the widget.
_typed   = st.chat_input("Ask me anything about Sowbhagya...")

# Chip injection: if a quick-action button was clicked, use that as the prompt.
# The chip value is consumed once and cleared immediately.
_injected = st.session_state.pop("_chip_prompt", None)

# Final prompt: chip wins over typed (chip fires on the rerun after button click)
prompt = _injected or _typed

if prompt:

    # Validate API key before proceeding
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is not set. "
            "Add it to `.env` locally, or to **Secrets** in Streamlit Community Cloud."
        )
        st.stop()

    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the LangGraph — thinking steps in status panel, response OUTSIDE
    with st.chat_message("assistant", avatar="🤖"):

        # ── Phase 1: run graph, show live thinking steps inside status ────────
        _status = st.status("Thinking...", expanded=True)

        response_text = ""
        last_agent = "rag_agent"
        _run_error = None
        result: dict | None = None

        try:
            graph = load_graph()

            lc_messages = [
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            _node_labels = {
                "supervisor":      ("🧠", "Routing your message..."),
                "rag_agent":       ("📚", "Searching knowledge base..."),
                "scheduler_agent": ("📅", "Opening scheduler..."),
                "notifier_agent":  ("📬", "Preparing notification..."),
                "chitchat_agent":  ("💬", "Composing reply..."),
            }

            _seen_nodes: list[str] = []
            _state_acc: dict = {
                "messages": lc_messages,
                "next_agent": "rag_agent",
                "knowledge_gap": False,
                "visitor_name": "",
                "visitor_email": "",
                "booking_state": {},
                "notified": False,
                "last_agent": st.session_state.last_agent,
            }

            # Stream: write thinking steps INSIDE the status box
            with _status:
                for chunk in graph.stream(
                    _state_acc.copy(),
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, delta in chunk.items():
                        if node_name not in _seen_nodes:
                            _seen_nodes.append(node_name)
                            _icon, _lbl = _node_labels.get(node_name, ("⚙️", f"Running {node_name}..."))
                            st.write(f"{_icon} **{_lbl}**")
                        for k, v in delta.items():
                            if k == "messages" and isinstance(v, list):
                                _state_acc["messages"] = _state_acc["messages"] + v
                            else:
                                _state_acc[k] = v

            result = _state_acc

            # Extract + normalise the final AI message text
            from agents.graph import _extract_text
            last_msg = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
                None,
            )
            response_text = _extract_text(
                last_msg.content if last_msg else "I'm not sure how to respond to that."
            )
            last_agent = result.get("last_agent", "rag_agent")

            # Save to session
            st.session_state.last_agent = last_agent
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text, "agent": last_agent}
            )
            _status.update(label="Done ✓", state="complete", expanded=False)

        except Exception as e:
            _status.update(label="Error", state="error", expanded=False)
            _run_error = e

        # ── Phase 2: render response OUTSIDE the status box ───────────────────
        if _run_error:
            is_quota, wait_secs = _is_quota_error(_run_error)
            if is_quota:
                st.session_state["_quota_hit"] = True
                _current = os.environ.get("GEMINI_MODEL", "models/gemini-3.5-flash")
                _rotation = [
                    "models/gemini-3.5-flash",
                    "models/gemini-3.5-flash-lite",
                    "models/gemini-3.1-flash-lite",
                    "models/gemini-3-flash-preview",
                    "models/gemini-2.5-flash",
                    "models/gemini-2.0-flash",
                    "models/gemini-2.0-flash-001",
                    "models/gemini-2.0-flash-lite",
                    "models/gemini-2.0-flash-lite-001",
                ]
                _next = next((m for m in _rotation if m != _current), "models/gemini-3.5-flash")
                st.warning(
                    f"**Quota exhausted** for `{_current.replace('models/', '')}`.  \n\n"
                    f"👉 Switch to **`{_next.replace('models/', '')}`** using the "
                    f"**🤖 Model** selector in the sidebar.  \n\n"
                    f"_Daily limits reset at midnight PT. Per-minute limits reset in ~{wait_secs}s._",
                    icon="⚠️",
                )
            else:
                st.error(f"Something went wrong: {_run_error}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(_run_error), "agent": "error"}
                )

        elif response_text:
            # Stream the response text word-by-word (typewriter effect)
            st.write_stream(_stream_text(response_text))

            # Agent tag (only for non-RAG agents)
            if last_agent != "rag_agent":
                _tag_labels = {
                    "scheduler_agent": "📅 Scheduler agent",
                    "notifier_agent":  "📬 Notifier agent",
                    "chitchat_agent":  "💬 Chit-chat agent",
                }
                st.markdown(
                    f'<div class="agent-tag">handled by {_tag_labels.get(last_agent, last_agent)}</div>',
                    unsafe_allow_html=True,
                )

            # Knowledge-gap banner
            if result is not None and result.get("knowledge_gap", False):
                st.info(
                    "💡 This question was outside my knowledge base. "
                    "Sowbhagya will be notified to answer directly.",
                    icon="📬",
                )
