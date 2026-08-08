"""
Sowbhagya's Personal AI — LangGraph Multi-Agent Graph
=======================================================
Architecture:
  Supervisor (Gemini) — intent classifier
      ├── RAG Agent        — retrieves from ChromaDB, answers as Sowbhagya
      ├── Scheduler Agent  — multi-turn meeting booking flow
      ├── Notifier Agent   — captures questions Sowbhagya can't answer here, sends email
      └── Chitchat Agent   — casual conversation as Sowbhagya

Each sub-agent is a fully autonomous ReAct agent with its own tools and system prompt.
The Supervisor routes intent and can re-route after each agent turn if needed.
State is persisted via InMemorySaver (zero-infra, works on Streamlit Cloud).
"""
from __future__ import annotations

import os
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from agents.prompts import (
    SUPERVISOR_SYSTEM,
    RAG_SYSTEM,
    SCHEDULER_SYSTEM,
    NOTIFIER_SYSTEM,
    CHITCHAT_SYSTEM,
)
from agents.tools import retrieve_profile_info, book_meeting, notify_owner


# ── Shared Graph State ────────────────────────────────────────────────────────

class TwinState(TypedDict):
    """
    Shared state across all nodes in the graph.

    messages        : full conversation history (append-only via operator.add)
    next_agent      : which sub-agent the supervisor chose to handle this turn
    knowledge_gap   : True when RAG couldn't find an answer — triggers notifier
    visitor_name    : captured during booking / notification flows
    visitor_email   : captured during booking / notification flows
    booking_state   : tracks progress through the multi-step booking flow
    notified        : True once owner notification has been sent (avoid duplicates)
    last_agent      : which agent handled the previous turn (for loopback logic)
    """
    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str
    knowledge_gap: bool
    visitor_name: str
    visitor_email: str
    booking_state: dict
    notified: bool
    last_agent: str


# ── LLM instance ──────────────────────────────────────────────────────────────
# Confirmed live models on this API key (via ListModels + live probe):
#
# Gemini 3.x  — fresh buckets, separate quotas, try these first:
#   gemini-3.5-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite
#   gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.6-flash
#
# Gemini 2.5  — 20 req/day only (use sparingly):
#   gemini-2.5-flash
#
# Gemini 2.0  — 1500 req/day each:
#   gemini-2.0-flash, gemini-2.0-flash-001, gemini-2.0-flash-lite, gemini-2.0-flash-lite-001
#
# NOT available: gemini-1.5-*, gemini-2.5-flash-lite, gemini-2.5-pro (quota 0)
#
# Override via GEMINI_MODEL env var or the sidebar model selector in app.py.

_DEFAULT_MODEL = "models/gemini-3.5-flash-lite"


def _llm() -> ChatGoogleGenerativeAI:
    """
    Returns a Gemini chat model.
    Override via GEMINI_MODEL env var or the sidebar model selector, e.g.:
        GEMINI_MODEL=models/gemini-2.5-flash streamlit run app.py
    """
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1,
        max_output_tokens=2048,   # raised from 512 — prevents mid-sentence cutoff
    )


# ── Safe content extractor ────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Normalise a LangChain message content to plain str.

    Gemini 2.5+ returns content as a list of parts:
      [{"type": "text", "text": "..."}, ...]
    Older models return a plain str.
    This helper handles both so every node can safely call .lower() / 'in' etc.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p)
    return str(content)


# ── Node: Supervisor ──────────────────────────────────────────────────────────

# ── keyword patterns for zero-token routing ───────────────────────────────────
_SCHEDULE_KW = {"book", "schedule", "meeting", "call", "appointment", "calendar", "connect", "interview"}
_NOTIFY_KW   = {"contact", "reach", "email", "message", "directly", "real person", "human"}
_CHAT_KW     = {"hi", "hello", "hey", "thanks", "thank you", "lol", "haha", "bye", "what's up", "sup"}


def supervisor_node(state: TwinState) -> dict:
    """
    Routes intent to the correct sub-agent.
    Free-tier optimisation: keyword matching first (zero tokens).
    LLM call only when keywords don't match — saves ~1 API call per turn.
    """
    last_human = _extract_text(
        next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
    ).lower()

    # Forward immediately if RAG already flagged a knowledge gap
    if state.get("knowledge_gap", False):
        return {"next_agent": "notifier_agent"}

    # Zero-cost keyword routing
    words = set(last_human.split())
    if any(kw in last_human for kw in _SCHEDULE_KW):
        return {"next_agent": "scheduler_agent"}
    if any(kw in last_human for kw in _NOTIFY_KW):
        return {"next_agent": "notifier_agent"}
    if words & _CHAT_KW and len(last_human) < 60:
        return {"next_agent": "chitchat_agent"}

    # Fallback: LLM routing for ambiguous messages
    routing_prompt = (
        f"{SUPERVISOR_SYSTEM}\n\n"
        f"Message: \"{last_human}\"\n"
        f"Reply with ONLY one word: rag_agent | scheduler_agent | notifier_agent | chitchat_agent"
    )
    response = _llm().invoke(
        [HumanMessage(content=routing_prompt)],
        config={"run_name": "supervisor-llm-routing"},
    )
    raw = _extract_text(response.content).strip().lower()

    if "scheduler" in raw:
        return {"next_agent": "scheduler_agent"}
    if "notifier" in raw:
        return {"next_agent": "notifier_agent"}
    if "chitchat" in raw:
        return {"next_agent": "chitchat_agent"}
    return {"next_agent": "rag_agent"}


# ── Node: RAG Agent ───────────────────────────────────────────────────────────

def build_rag_agent():
    """ReAct agent: retrieves from ChromaDB, answers profile questions."""
    return create_react_agent(
        model=_llm(),
        tools=[retrieve_profile_info],
        prompt=RAG_SYSTEM,
    )


def rag_node(state: TwinState) -> dict:
    """Runs the RAG ReAct agent and detects knowledge gaps."""
    agent = build_rag_agent()
    result = agent.invoke(
        {"messages": state["messages"]},
        config={"run_name": "rag-agent"},
    )

    last_ai = result["messages"][-1]
    response_text = _extract_text(
        last_ai.content if isinstance(last_ai, AIMessage) else str(last_ai)
    )

    # Detect if the agent signalled it could not answer
    rt_lower = response_text.lower()
    knowledge_gap = (
        "knowledge_gap" in rt_lower
        or "don't have that detail" in rt_lower
        or "flag this for sowbhagya" in rt_lower
        or "KNOWLEDGE_GAP" in response_text
    )

    return {
        "messages": [AIMessage(content=response_text, name="rag_agent")],
        "knowledge_gap": knowledge_gap,
        "last_agent": "rag_agent",
    }


# ── Node: Scheduler Agent ─────────────────────────────────────────────────────

def build_scheduler_agent():
    """Multi-turn booking ReAct agent."""
    return create_react_agent(
        model=_llm(),
        tools=[book_meeting],
        prompt=SCHEDULER_SYSTEM,
    )


def scheduler_node(state: TwinState) -> dict:
    """Runs the scheduler ReAct agent."""
    agent = build_scheduler_agent()
    result = agent.invoke(
        {"messages": state["messages"]},
        config={"run_name": "scheduler-agent"},
    )

    last_ai = result["messages"][-1]
    response_text = _extract_text(
        last_ai.content if isinstance(last_ai, AIMessage) else str(last_ai)
    )

    booking_update = state.get("booking_state", {})
    if "BOOKING_CONFIRMED" in response_text:
        booking_update["confirmed"] = True

    return {
        "messages": [AIMessage(content=response_text, name="scheduler_agent")],
        "booking_state": booking_update,
        "knowledge_gap": False,
        "last_agent": "scheduler_agent",
    }


# ── Node: Notifier Agent ──────────────────────────────────────────────────────

def build_notifier_agent():
    """Escalation ReAct agent — collects contact info, fires notify_owner."""
    return create_react_agent(
        model=_llm(),
        tools=[notify_owner],
        prompt=NOTIFIER_SYSTEM,
    )


def notifier_node(state: TwinState) -> dict:
    """Runs the notifier ReAct agent."""
    agent = build_notifier_agent()
    result = agent.invoke(
        {"messages": state["messages"]},
        config={"run_name": "notifier-agent"},
    )

    last_ai = result["messages"][-1]
    response_text = _extract_text(
        last_ai.content if isinstance(last_ai, AIMessage) else str(last_ai)
    )

    notified = state.get("notified", False)
    if "NOTIFICATION_SENT" in response_text or "NOTIFICATION_LOGGED" in response_text:
        notified = True

    return {
        "messages": [AIMessage(content=response_text, name="notifier_agent")],
        "knowledge_gap": False,
        "notified": notified,
        "last_agent": "notifier_agent",
    }


# ── Node: Chit-Chat Agent ─────────────────────────────────────────────────────

def chitchat_node(state: TwinState) -> dict:
    """Lightweight personality agent — small talk, no tools, minimal tokens."""
    messages = [SystemMessage(content=CHITCHAT_SYSTEM)] + state["messages"]
    response = _llm().invoke(
        messages,
        config={"run_name": "chitchat-agent"},
    )
    return {
        "messages": [AIMessage(content=_extract_text(response.content), name="chitchat_agent")],
        "knowledge_gap": False,
        "last_agent": "chitchat_agent",
    }


# ── Routing: after supervisor decides ─────────────────────────────────────────

def route_after_supervisor(
    state: TwinState,
) -> Literal["rag_agent", "scheduler_agent", "notifier_agent", "chitchat_agent"]:
    return state["next_agent"]


# ── Routing: after RAG agent ──────────────────────────────────────────────────

def route_after_rag(state: TwinState) -> Literal["notifier_agent", END]:
    """If RAG detected a knowledge gap, immediately forward to notifier."""
    if state.get("knowledge_gap", False):
        return "notifier_agent"
    return END


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> "CompiledGraph":
    """
    Assembles and compiles Sowbhagya's personal AI LangGraph.

    Graph flow:
      START → supervisor → [rag | scheduler | notifier | chitchat]
                                   ↓ (if knowledge_gap)
                               notifier → END
    """
    builder = StateGraph(TwinState)

    # Register all nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_node)
    builder.add_node("scheduler_agent", scheduler_node)
    builder.add_node("notifier_agent", notifier_node)
    builder.add_node("chitchat_agent", chitchat_node)

    # Entry: always start at supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor routes to one of four agents
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "rag_agent": "rag_agent",
            "scheduler_agent": "scheduler_agent",
            "notifier_agent": "notifier_agent",
            "chitchat_agent": "chitchat_agent",
        },
    )

    # RAG can forward to notifier on knowledge gap
    builder.add_conditional_edges(
        "rag_agent",
        route_after_rag,
        {
            "notifier_agent": "notifier_agent",
            END: END,
        },
    )

    # Scheduler, notifier, chitchat always terminate the turn
    builder.add_edge("scheduler_agent", END)
    builder.add_edge("notifier_agent", END)
    builder.add_edge("chitchat_agent", END)

    # InMemorySaver — persists conversation within a single Streamlit process.
    # Each Streamlit Cloud deployment gets its own process so this is sufficient.
    # For true cross-session persistence, swap with AsyncSqliteSaver or a Postgres saver.
    memory = InMemorySaver()

    return builder.compile(checkpointer=memory)


# ── Convenience: get/cache the compiled graph ─────────────────────────────────

_graph_instance = None


def get_graph() -> "CompiledGraph":
    """Returns a singleton compiled graph (one per process, cached via @st.cache_resource)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


# ── Initial state factory ─────────────────────────────────────────────────────

def initial_state() -> TwinState:
    return TwinState(
        messages=[],
        next_agent="rag_agent",
        knowledge_gap=False,
        visitor_name="",
        visitor_email="",
        booking_state={},
        notified=False,
        last_agent="none",
    )
