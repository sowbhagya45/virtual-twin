"""
app.py — virtual-twin Streamlit Application
"""
from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def _load_secrets():
    """Bridge Streamlit secrets → os.environ.

    Covers all keys that must be in the environment before any import
    of agents/ or langchain runs (tracing vars are read at import time).
    """
    _KEYS = (
        "GOOGLE_API_KEY",
        "GMAIL_USER",
        "GMAIL_APP_PWD",
        "OWNER_EMAIL",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_TRACING_V2",   # must be set before langchain imports
        "LANGCHAIN_PROJECT",
        "GEMINI_MODEL",
    )
    try:
        for key in _KEYS:
            if key not in os.environ and key in st.secrets:
                os.environ[key] = st.secrets[key]
    except Exception:
        pass

_load_secrets()


def _setup_langsmith() -> bool:
    """Activate LangSmith tracing if LANGSMITH_API_KEY is present.

    The LangSmith SDK reads LANGCHAIN_API_KEY (not LANGSMITH_API_KEY) plus
    LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT.  We always write them
    unconditionally (not setdefault) so a stale value from a previous run
    cannot shadow the current key.

    Returns True if tracing is enabled, False if the key is missing.
    """
    api_key = (
        os.environ.get("LANGSMITH_API_KEY")
        or os.environ.get("LANGCHAIN_API_KEY")   # accept either name
        or ""
    )
    if not api_key:
        return False
    # LangSmith SDK requires LANGCHAIN_API_KEY — always overwrite so it's fresh
    os.environ["LANGCHAIN_API_KEY"]      = api_key
    os.environ["LANGCHAIN_TRACING_V2"]   = "true"
    os.environ["LANGCHAIN_PROJECT"]      = os.environ.get(
        "LANGCHAIN_PROJECT", "sowbhagya-personal-ai"
    )
    return True

_tracing_enabled = _setup_langsmith()

def _is_quota_error(exc: Exception) -> tuple[bool, int]:
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        import re
        m = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
        return True, int(m.group(1)) if m else 60
    return False, 0

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sowbhagya Mohanthy",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 0 !important; }

/* ── Dark sidebar ── */
[data-testid="stSidebar"] > div:first-child {
    background: #111827 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.st-emotion-cache-1sno8jx),
[data-testid="stSidebar"] div:not([data-baseweb]),
[data-testid="stSidebar"] label { color: #9ca3af !important; }
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] h3 { color: #f9fafb !important; }
[data-testid="stSidebar"] a { color: #a78bfa !important; text-decoration: none; }
[data-testid="stSidebar"] a:hover { color: #c4b5fd !important; }
[data-testid="stSidebar"] hr { border-color: #374151 !important; margin: 12px 0 !important; }

/* Sidebar buttons (Download + Clear) — override the secondary chip styles */
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[kind="secondary"] {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    color: #d1d5db !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #374151 !important;
    color: #f9fafb !important;
    border-color: #4b5563 !important;
}

/* Sidebar select / model picker — control box */
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] * {
    color: #d1d5db !important;
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"]:focus-within {
    border-color: #6d28d9 !important;
    box-shadow: 0 0 0 2px rgba(109,40,217,0.2) !important;
}

/* Model dropdown list — dark theme (renders at document root via fixed positioning) */
[data-testid="stSelectboxVirtualDropdown"] {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
    overflow: hidden !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    background: transparent !important;
    color: #d1d5db !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
    background: #374151 !important;
    color: #f9fafb !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:focus {
    background: #374151 !important;
    color: #f9fafb !important;
    outline: none !important;
}

/* Sidebar expander (Switch model) */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    background: #1f2937 !important;
    margin-bottom: 8px !important;
    overflow: hidden !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"]:focus-within {
    border-color: #374151 !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: #1f2937 !important;
    color: #a78bfa !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: #374151 !important;
    color: #c4b5fd !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg { color: #a78bfa !important; }
/* Expander content area */
[data-testid="stSidebar"] [data-testid="stExpander"] > div:last-child {
    background: #1f2937 !important;
    border-top: 1px solid #374151 !important;
    padding: 10px 12px !important;
}

/* ── Chip / quick-action buttons — all identical outline style ── */
button[kind="primary"],
button[kind="secondary"] {
    background: #fff !important;
    border: 1.5px solid #e5e7eb !important;
    color: #374151 !important;
    border-radius: 9999px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
    background: #f5f3ff !important;
    border-color: #a78bfa !important;
    color: #6d28d9 !important;
}

/* ── Chat messages ── */
/* Bot avatar (exttvjz2) — purple gradient */
.exttvjz2 {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 0 !important;        /* hide raw 🤖 emoji text */
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.exttvjz2::after {
    content: "✦";
    font-size: 15px;
    color: #fff;
    line-height: 1;
}

/* User avatar (exttvjz3) — pink-orange gradient */
.exttvjz3 {
    background: linear-gradient(135deg, #ec4899 0%, #f97316 100%) !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 0 !important;        /* hide raw icon text */
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.exttvjz3::after {
    content: "◉";
    font-size: 14px;
    color: #fff;
    line-height: 1;
}

/* Bot message bubble — uses exttvjz2 as the anchor */
[data-testid="stChatMessage"]:has(.exttvjz2) [data-testid="stChatMessageContent"] {
    background: #f8f7ff !important;
    border-radius: 4px 14px 14px 14px !important;
    padding: 14px 18px !important;
    font-size: 14px;
    line-height: 1.75;
    border: 1px solid #ede9fe !important;
    color: #1f2328 !important;
}
/* User message bubble */
[data-testid="stChatMessage"]:has(.exttvjz3) [data-testid="stChatMessageContent"] {
    background: #7c3aed !important;
    border-radius: 14px 14px 4px 14px !important;
    padding: 12px 18px !important;
    border: none !important;
}
/* Ensure user text is white */
[data-testid="stChatMessage"]:has(.exttvjz3) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has(.exttvjz3) [data-testid="stChatMessageContent"] p {
    color: #fff !important;
    font-size: 14px;
}

/* ── "Done ✓" / thinking status expander inside chat ── */
[data-testid="stChatMessage"] [data-testid="stExpander"] {
    background: #f0edff !important;
    border: 1px solid #ede9fe !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    margin-bottom: 6px !important;
}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary {
    background: #f0edff !important;
    color: #6d28d9 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary:hover {
    background: #ede9fe !important;
}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary svg,
[data-testid="stChatMessage"] [data-testid="stExpander"] summary span {
    color: #7c3aed !important;
}

/* Chat spacing — tighter between messages */
[data-testid="stChatMessage"] {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}

/* ── Chat input — single-line pill ── */
[data-testid="stChatInput"] {
    background: #fff !important;
    border-top: 1px solid #f3f4f6 !important;
}
[data-testid="stChatInput"] .e1p9v2yr1 {
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 8px 10px 8px 16px !important;
    background: #f9fafb !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] .e1p9v2yr1:focus-within {
    border-color: #6d28d9 !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(109,40,217,0.10) !important;
}
[data-testid="stChatInput"] .e1p9v2yr3 {
    flex-direction: row !important;
    align-items: center !important;
    flex: 1 !important;
    height: auto !important;
    gap: 8px !important;
}
[data-testid="stChatInput"] .e1p9v2yr4 {
    flex: 1 !important;
    height: auto !important;
}
[data-testid="stChatInput"] .e1p9v2yr7 { display: none !important; }
[data-testid="stChatInput"] textarea {
    border: none !important;
    background: transparent !important;
    font-size: 14px !important;
    color: #1f2328 !important;
    padding: 4px 0 !important;
    resize: none !important;
    outline: none !important;
    box-shadow: none !important;
    min-height: 28px !important;
    max-height: 120px !important;
    line-height: 1.5 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9ca3af !important; }
[data-testid="stChatInput"] .e1p9v2yr6 {
    display: flex !important;
    align-items: center !important;
    flex-shrink: 0 !important;
}
/* Send button — always visible, dims when disabled */
[data-testid="stChatInput"] button {
    background: #6d28d9 !important;
    border-radius: 10px !important;
    border: none !important;
    flex-shrink: 0 !important;
    width: 36px !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s, opacity 0.15s !important;
}
[data-testid="stChatInput"] button:hover:not(:disabled) { background: #5b21b6 !important; }
[data-testid="stChatInput"] button:disabled {
    background: #c4b5fd !important;
    cursor: default !important;
    opacity: 1 !important;
}
/* Always-white arrow icon regardless of disabled state */
[data-testid="stChatInput"] button svg,
[data-testid="stChatInput"] button svg path,
[data-testid="stChatInput"] button span { color: #fff !important; fill: #fff !important; }

/* ── Bottom container — remove dead space ── */
[data-testid="stBottomBlockContainer"] {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

/* ── Welcome page spacer — reduce oversized gap ── */
.welcome-spacer { height: 8vh !important; }

/* ── Quick chips gap ── */
.stHorizontalBlock { margin-bottom: 4px !important; }

/* ── Sidebar «/» collapse button inside sidebar header ── */
[data-testid="stBaseButton-headerNoPadding"] {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    visibility: visible !important;
    transition: background 0.15s, border-color 0.15s !important;
}
[data-testid="stBaseButton-headerNoPadding"]:hover {
    background: #374151 !important;
    border-color: #4b5563 !important;
}
[data-testid="stBaseButton-headerNoPadding"] svg {
    color: #a78bfa !important;
    fill: #a78bfa !important;
    stroke: #a78bfa !important;
}
[data-testid="stBaseButton-headerNoPadding"]:hover svg {
    color: #c4b5fd !important;
    fill: #c4b5fd !important;
    stroke: #c4b5fd !important;
}

/* Ensure header doesn't clip the button */
header[data-testid="stHeader"] {
    overflow: visible !important;
    z-index: 999 !important;
}

/* ── Sidebar collapsed expand tab ── */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-left: none !important;
    border-radius: 0 8px 8px 0 !important;
    width: 24px !important;
    min-height: 48px !important;
    padding: 0 !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover {
    background: #374151 !important;
    border-color: #4b5563 !important;
}
[data-testid="stSidebarCollapsedControl"] button svg {
    color: #a78bfa !important;
    fill: #a78bfa !important;
    stroke: #a78bfa !important;
}

/* ── Agent tag ── */
.agent-tag {
    font-size: 11px;
    color: #a78bfa;
    font-style: italic;
    margin-top: 6px;
    padding-left: 2px;
}

/* ── Error / warning alert in chat ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 13.5px !important;
}

/* ── Status widget (Thinking…) ── */
[data-testid="stStatusContainer"] {
    border-radius: 10px !important;
    border: 1px solid #ede9fe !important;
}
</style>
""", unsafe_allow_html=True)

# ── Graph bootstrap ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading agents…")
def load_graph():
    if not os.path.exists("./chroma_db"):
        st.toast("Building knowledge base — ~2 min…", icon="⚙️")
        from ingest import ingest
        ingest()
    from agents.graph import get_graph
    return get_graph()

# ── Model options ──────────────────────────────────────────────────────────────
_MODEL_OPTIONS = {
    "gemini-3.5-flash  [3.x ✦]":       "models/gemini-3.5-flash",
    "gemini-3.5-flash-lite  [3.x ✦]":  "models/gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite  [3.x ✦]":  "models/gemini-3.1-flash-lite",
    "gemini-2.5-flash  [2.5 — 20/day]": "models/gemini-2.5-flash",
    "gemini-2.0-flash  [2.0]":          "models/gemini-2.0-flash",
    "gemini-2.0-flash-lite  [2.0]":     "models/gemini-2.0-flash-lite",
}
_cur_env   = os.environ.get("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
_cur_label = next((k for k, v in _MODEL_OPTIONS.items() if v == _cur_env), list(_MODEL_OPTIONS)[1])

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sowbhagya Mohanthy")
    st.markdown(
        "AI Engineer & Software Developer  \n"
        "📍 IBM · Hyderabad  \n"
        "_Ask me anything — I'll answer directly._",
    )
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white&style=flat-square)](https://www.linkedin.com/in/sowbhagya-mohanthy-8a8a68221) "
        "[![HuggingFace](https://img.shields.io/badge/🤗_HF-FFD21F?style=flat-square)](https://huggingface.co/Sowbhagya-45)"
    )
    st.divider()

    # Model
    _short = _cur_env.replace("models/", "")
    st.caption(f"🟢 {_short} · active")
    with st.expander("Switch model"):
        _sel = st.selectbox("", list(_MODEL_OPTIONS.keys()),
                            index=list(_MODEL_OPTIONS.keys()).index(_cur_label),
                            label_visibility="collapsed")
        if _MODEL_OPTIONS[_sel] != _cur_env:
            os.environ["GEMINI_MODEL"] = _MODEL_OPTIONS[_sel]
            st.session_state.pop("_quota_hit", None)
            load_graph.clear()
            st.rerun()

    if st.session_state.get("_quota_hit"):
        st.error("Quota hit — switch model above", icon="⚠️")

    # Resume
    _resume = os.path.join("data", "resume.pdf")
    if os.path.exists(_resume):
        with open(_resume, "rb") as _f:
            st.download_button("📄 Download Resume", _f,
                               file_name="Sowbhagya_Mohanthy_Resume.pdf",
                               mime="application/pdf", use_container_width=True)
    # Clear
    if st.button("🗑 Clear chat", use_container_width=True):
        for k in ("messages", "thread_id", "last_agent"):
            st.session_state.pop(k, None)
        st.rerun()

    st.caption("Powered by LangGraph + Gemini")

# ── Sidebar toggle floating button (JS via components) ────────────────────────
import streamlit.components.v1 as _components
_components.html("""
<script>
(function() {
    function injectToggleBtn() {
        var root = window.parent.document;
        if (root.getElementById('_sb_toggle_fab')) return;
        var btn = root.createElement('button');
        btn.id = '_sb_toggle_fab';
        btn.title = 'Toggle sidebar';
        // Panel icon SVG — cleaner than ☰
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="1" y="1" width="14" height="14" rx="2" stroke="#a78bfa" stroke-width="1.5"/><line x1="5" y1="1" x2="5" y2="15" stroke="#a78bfa" stroke-width="1.5"/></svg>';
        btn.style.cssText = [
            'position:fixed','top:10px','left:10px','z-index:9999',
            'width:32px','height:32px','border-radius:8px',
            'background:#1f2937','border:1px solid #374151',
            'cursor:pointer','display:flex','align-items:center',
            'justify-content:center',
            'transition:background 0.15s,border-color 0.15s',
            'box-shadow:0 2px 8px rgba(0,0,0,0.35)'
        ].join(';');
        btn.onmouseenter = function(){
            btn.style.background='#374151';
            btn.style.borderColor='#4b5563';
            btn.style.opacity='1';
            // lighten SVG strokes on hover
            btn.querySelectorAll('rect,line').forEach(function(el){
                el.setAttribute('stroke','#c4b5fd');
            });
        };
        btn.onmouseleave = function(){
            btn.style.background='#1f2937';
            btn.style.borderColor='#374151';
            btn.style.opacity='0.85';
            btn.querySelectorAll('rect,line').forEach(function(el){
                el.setAttribute('stroke','#a78bfa');
            });
        };
        btn.onclick = function() {
            var native = root.querySelector('[data-testid="stBaseButton-headerNoPadding"]')
                      || root.querySelector('[data-testid="stSidebarCollapsedControl"] button')
                      || root.querySelector('button[aria-label*="sidebar"]')
                      || root.querySelector('button[aria-label*="Sidebar"]');
            if (native) { native.click(); }
        };
        root.body.appendChild(btn);
    }
    if (window.parent.document.readyState === 'loading') {
        window.parent.document.addEventListener('DOMContentLoaded', injectToggleBtn);
    } else {
        injectToggleBtn();
    }
})();
</script>
""", height=0)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "last_agent" not in st.session_state: st.session_state.last_agent = "none"

# ── Streaming helper ───────────────────────────────────────────────────────────
import time as _time
def _stream(text: str, delay: float = 0.012):
    for word in text.split(" "):
        yield word + " "
        _time.sleep(delay)

# ── Welcome + quick chips (empty state only) ───────────────────────────────────
_CHIPS = [
    ("🏆 Key skills",   "What are your key skills?"),
    ("🚀 Top project",  "Tell me about your flagship project"),
    ("📅 schedule a connect",  "I'd like to schedule a meeting with you"),
    ("📩 Contact",      "I want to reach out to you directly"),
]

if not st.session_state.messages:
    # Vertical centering spacer — pushes welcome block to middle of viewport
    st.markdown(
        '<div class="welcome-spacer" style="height:8vh"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Hey, I'm Sowbhagya 👋")
    st.markdown(
        "Ask me anything about my background, skills, or projects. "
        "You can also **schedule a connect** with me or leave a message."
    )
    st.write("")

    # Chip buttons — simple columns
    cols = st.columns(len(_CHIPS))
    for col, (label, prompt) in zip(cols, _CHIPS):
        with col:
            if st.button(label, use_container_width=True,
                         type="primary" if label.startswith("🏆") else "secondary",
                         key=f"chip_{label}"):
                st.session_state["_pending"] = prompt
else:
    # Compact chip row when conversation exists
    cols = st.columns(len(_CHIPS))
    for col, (label, prompt) in zip(cols, _CHIPS):
        with col:
            if st.button(label, use_container_width=True,
                         type="primary" if label.startswith("🏆") else "secondary",
                         key=f"chip2_{label}"):
                st.session_state["_pending"] = prompt
    st.write("")

# ── Render history ─────────────────────────────────────────────────────────────
_ICON = {"rag_agent": "🤖", "scheduler_agent": "📅", "notifier_agent": "📬", "chitchat_agent": "💬"}
_TAG  = {"scheduler_agent": "📅 Scheduler", "notifier_agent": "📬 Notifier", "chitchat_agent": "💬 Chit-chat"}

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar=_ICON.get(msg.get("agent", "rag_agent"), "🤖")):
            st.markdown(msg["content"])
            if msg.get("agent") and msg["agent"] not in ("rag_agent", "error"):
                st.markdown(f'<div class="agent-tag">via {_TAG.get(msg["agent"], msg["agent"])}</div>',
                            unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
_typed  = st.chat_input("Ask me anything…")
prompt  = st.session_state.pop("_pending", None) or _typed

if prompt:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not set. Add it to `.env` or Streamlit Secrets.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        _status = st.status("Thinking…", expanded=True)
        response_text, last_agent, _err, result = "", "rag_agent", None, None

        try:
            graph = load_graph()
            lc_msgs = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]
            _model_short = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").replace("models/", "")
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                # ── LangSmith trace metadata ──────────────────────────────────
                # run_name   → top-level run name shown in the trace list
                # tags       → filterable labels in LangSmith UI
                # metadata   → key-value pairs visible inside each trace
                "run_name": f"chat | {prompt[:60]}",
                "tags": ["sowbhagya-ai", _model_short],
                "metadata": {
                    "thread_id":   st.session_state.thread_id,
                    "model":       _model_short,
                    "turn":        len(st.session_state.messages),
                    "user_prompt": prompt[:200],
                },
            }
            _NODE = {
                "supervisor":      ("🧠", "Planning…"),
                "rag_agent":       ("📚", "Searching knowledge base…"),
                "scheduler_agent": ("📅", "Opening scheduler…"),
                "notifier_agent":  ("📬", "Sending notification…"),
                "chitchat_agent":  ("💬", "Composing reply…"),
            }
            _seen: list[str] = []
            _state: dict = {
                "messages":      lc_msgs,
                "plan":          [],
                "plan_step":     0,
                "rag_output":    "",
                "knowledge_gap": False,
                "visitor_name":  "",
                "visitor_email": "",
                "booking_state": {},
                "notified":      False,
                "last_agent":    st.session_state.last_agent,
            }
            with _status:
                for chunk in graph.stream(_state.copy(), config=config, stream_mode="updates"):
                    for node, delta in chunk.items():
                        if node not in _seen:
                            _seen.append(node)
                            ic, lb = _NODE.get(node, ("⚙️", f"Running {node}…"))
                            # Show plan after supervisor decides
                            plan_label = ""
                            if node == "supervisor" and "plan" in delta:
                                plan = delta["plan"]
                                if len(plan) > 1:
                                    plan_label = f" → `{'` → `'.join(plan)}`"
                            st.write(f"{ic} **{lb}**{plan_label}")
                        for k, v in delta.items():
                            if k == "messages" and isinstance(v, list):
                                _state["messages"] = _state["messages"] + v
                            else:
                                _state[k] = v
            result = _state

            from agents.graph import _extract_text
            last_msg = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
            response_text = _extract_text(last_msg.content if last_msg else "I'm not sure how to respond.")
            last_agent = result.get("last_agent", "rag_agent")

            st.session_state.last_agent = last_agent
            st.session_state.messages.append({"role": "assistant", "content": response_text, "agent": last_agent})
            _status.update(label="Done ✓", state="complete", expanded=False)

        except Exception as e:
            _status.update(label="Error", state="error", expanded=False)
            _err = e

        if _err:
            is_quota, wait = _is_quota_error(_err)
            if is_quota:
                st.session_state["_quota_hit"] = True
                _cur = os.environ.get("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
                _rot = ["models/gemini-3.5-flash", "models/gemini-3.5-flash-lite",
                        "models/gemini-3.1-flash-lite", "models/gemini-2.5-flash",
                        "models/gemini-2.0-flash"]
                _nxt = next((m for m in _rot if m != _cur), _rot[0])
                st.warning(
                    f"**Quota exhausted** for `{_cur.replace('models/','')}`.  \n"
                    f"Switch to **`{_nxt.replace('models/','')}`** in the sidebar.  \n"
                    f"_Resets at midnight PT. Per-minute resets in ~{wait}s._", icon="⚠️"
                )
            else:
                st.error(f"Something went wrong: {_err}")
                st.session_state.messages.append({"role": "assistant", "content": str(_err), "agent": "error"})

        elif response_text:
            st.write_stream(_stream(response_text))
            if last_agent not in ("rag_agent", "error"):
                st.markdown(f'<div class="agent-tag">via {_TAG.get(last_agent, last_agent)}</div>',
                            unsafe_allow_html=True)
            if result and result.get("knowledge_gap"):
                st.info("💡 Outside my knowledge — Sowbhagya will be notified.", icon="📬")
