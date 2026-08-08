"""
Sowbhagya's Personal AI — LangGraph Agentic Graph
===================================================
Architecture (v2 — true agentic loop):
  Supervisor (Planner) — decides a SEQUENCE of agents, not just one.
      Plans are lists like ["rag_agent", "notifier_agent"] for multi-step tasks.

  Plan Executor — runs each agent in the plan in order, threading outputs
      as context into the next agent.

  Sub-agents:
      ├── RAG Agent        — retrieves from ChromaDB, answers as Sowbhagya
      ├── Scheduler Agent  — multi-turn meeting booking flow
      ├── Notifier Agent   — two modes:
      │       COLLECT mode  → ask for visitor contact info, send alert to Sowbhagya
      │       SEND mode     → receives pre-fetched content, emails it to visitor
      └── Chitchat Agent   — casual conversation as Sowbhagya

Flow:
  START → supervisor (plan) → plan_executor loop → END
  The executor runs agents[plan[step]], increments step, loops until plan exhausted.

Example plans:
  "Tell me about your skills"            → ["rag_agent"]
  "Send me your details"                 → ["rag_agent", "notifier_agent"]
  "Book a meeting"                       → ["scheduler_agent"]
  "I want to connect with you"           → ["notifier_agent"]
  "Hi!"                                  → ["chitchat_agent"]
  "I can't find the answer"  (RAG gap)   → ["notifier_agent"]   (supervisor re-plans)
"""
from __future__ import annotations

import json
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
    SCHEDULER_SYSTEM_TEMPLATE,
    NOTIFIER_COLLECT_SYSTEM,
    NOTIFIER_SEND_SYSTEM,
    CHITCHAT_SYSTEM,
)
from agents.tools import retrieve_profile_info, send_email, create_calendar_event


# ── Shared Graph State ────────────────────────────────────────────────────────

class TwinState(TypedDict):
    """
    Shared state across all nodes in the graph.

    messages        : full conversation history (append-only via operator.add)
    plan            : ordered list of agent names to execute this turn
                      e.g. ["rag_agent", "notifier_agent"]
    plan_step       : index into plan — which agent runs next
    rag_output      : text result from the RAG agent (passed to notifier SEND mode)
    knowledge_gap   : True when RAG couldn't find an answer — triggers re-plan
    visitor_name    : captured during booking / notification flows
    visitor_email   : captured during booking / notification flows
    booking_state   : tracks progress through the multi-step booking flow
    notified        : True once owner notification has been sent (avoid duplicates)
    last_agent      : which agent handled the previous turn (for UI display)
    """
    messages:      Annotated[list[BaseMessage], operator.add]
    plan:          list[str]
    plan_step:     int
    rag_output:    str
    knowledge_gap: bool
    visitor_name:  str
    visitor_email: str
    booking_state: dict
    notified:      bool
    last_agent:    str


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
        max_output_tokens=2048,
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


# ── Node: Supervisor (Planner) ────────────────────────────────────────────────
#
# The supervisor ALWAYS asks the LLM to plan — no keyword fast-paths.
# The LLM returns a JSON array of agent names in execution order.
# The plan can be 1, 2, 3, or 4 agents long — the executor handles any length.
#
# Only exception: if a knowledge_gap was flagged by RAG mid-plan, we inject
# notifier_agent at the current plan position (interrupt + escalate).

_VALID_AGENTS = {"rag_agent", "scheduler_agent", "notifier_agent", "chitchat_agent"}


def _plan(last_human: str, context: str = "") -> list[str]:
    """
    Ask the LLM to produce an ordered execution plan for the visitor's message.

    Returns a deduplicated list of valid agent names.
    Falls back to ["rag_agent"] on any parse or validation failure.

    `context` is injected when re-planning mid-turn (e.g. after a knowledge gap)
    so the LLM knows what has already been attempted.
    """
    context_block = f"\nContext so far:\n{context}\n" if context else ""
    prompt = (
        f"{SUPERVISOR_SYSTEM}\n"
        f"{context_block}\n"
        f"Visitor message: \"{last_human}\"\n\n"
        "Reply with a JSON array of agent names in execution order.\n"
        "Available agents: rag_agent, scheduler_agent, notifier_agent, chitchat_agent\n"
        "\n"
        "Rules:\n"
        "  - Include every agent needed to FULLY satisfy the request.\n"
        "  - Order matters: if RAG content must be emailed, RAG comes before notifier.\n"
        "  - If booking AND sending details are both requested, include all three:\n"
        '    ["rag_agent", "notifier_agent", "scheduler_agent"]\n'
        "  - Do not repeat the same agent twice.\n"
        "  - Respond with ONLY the JSON array, nothing else."
    )
    try:
        response = _llm().invoke(
            [HumanMessage(content=prompt)],
            config={"run_name": "supervisor-plan"},
        )
        raw = _extract_text(response.content).strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, list):
            return ["rag_agent"]
        # Validate, deduplicate, preserve order
        seen, result = set(), []
        for agent in parsed:
            if agent in _VALID_AGENTS and agent not in seen:
                seen.add(agent)
                result.append(agent)
        return result or ["rag_agent"]
    except Exception:
        return ["rag_agent"]


def supervisor_node(state: TwinState) -> dict:
    """
    Plans which agents to run this turn, in order.
    Always calls the LLM — no keyword shortcuts.

    Returns {"plan": [...], "plan_step": 0}

    Special case: if RAG flagged a knowledge_gap mid-plan, inject notifier_agent
    at the current position so the visitor gets escalated immediately.
    """
    last_human = _extract_text(
        next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
    )

    # ── Re-plan after a knowledge gap ────────────────────────────────────────
    # RAG ran but couldn't answer — splice notifier_agent in at current step.
    if state.get("knowledge_gap", False):
        existing_plan = state.get("plan", [])
        step = state.get("plan_step", 0)
        # Keep any remaining planned agents after notifier (e.g. scheduler still pending)
        remaining = [a for a in existing_plan[step:] if a != "notifier_agent"]
        new_plan = ["notifier_agent"] + remaining
        return {
            "plan":          new_plan,
            "plan_step":     0,
            "knowledge_gap": False,
        }

    # ── LLM plans every turn ─────────────────────────────────────────────────
    plan = _plan(last_human)
    return {"plan": plan, "plan_step": 0}


# ── Helper: extract only the human conversation (no inter-agent AI messages) ──

def _human_messages(state: TwinState) -> list[BaseMessage]:
    """
    Return only the messages that came from the human visitor.

    Gemini's ReAct loop requires the last message to be a HumanMessage or
    ToolMessage.  When agents run in sequence, each previous agent appends an
    AIMessage to state["messages"].  If the next agent receives those, Gemini
    raises "Model does not support model prefilling."

    Solution: each agent is only given the original human turns.  The inter-
    agent outputs live in dedicated state fields (rag_output, booking_state…)
    and are injected via the system prompt, not as conversation history.
    """
    return [m for m in state["messages"] if isinstance(m, HumanMessage)]


# ── Node: RAG Agent ───────────────────────────────────────────────────────────

def rag_node(state: TwinState) -> dict:
    """
    Runs the RAG ReAct agent.
    Stores its output in rag_output so a subsequent notifier_agent in SEND mode
    can use it to email the content to the visitor.
    """
    agent = create_react_agent(
        model=_llm(),
        tools=[retrieve_profile_info],
        prompt=RAG_SYSTEM,
    )
    result = agent.invoke(
        {"messages": _human_messages(state)},
        config={"run_name": "rag-agent"},
    )

    last_ai = result["messages"][-1]
    response_text = _extract_text(
        last_ai.content if isinstance(last_ai, AIMessage) else str(last_ai)
    )

    rt_lower = response_text.lower()
    knowledge_gap = (
        "knowledge_gap" in rt_lower
        or "KNOWLEDGE_GAP" in response_text
        or "don't have that detail" in rt_lower
        or "reach out and i'll answer" in rt_lower
    )

    return {
        "messages":      [AIMessage(content=response_text, name="rag_agent")],
        "rag_output":    response_text,
        "knowledge_gap": knowledge_gap,
        "plan_step":     state.get("plan_step", 0) + 1,
        "last_agent":    "rag_agent",
    }


# ── Node: Scheduler Agent ─────────────────────────────────────────────────────

def scheduler_node(state: TwinState) -> dict:
    """Multi-turn meeting booking ReAct agent — creates real Google Calendar events."""
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")  # e.g. "2026-08-09 11:42"
    prompt = SystemMessage(
        content=SCHEDULER_SYSTEM_TEMPLATE.replace("{{NOW}}", now_str)
    )
    agent = create_react_agent(
        model=_llm(),
        tools=[create_calendar_event],
        prompt=prompt,
    )
    result = agent.invoke(
        {"messages": _human_messages(state)},
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
        "messages":      [AIMessage(content=response_text, name="scheduler_agent")],
        "booking_state": booking_update,
        "knowledge_gap": False,
        "plan_step":     state.get("plan_step", 0) + 1,
        "last_agent":    "scheduler_agent",
    }


# ── Node: Notifier Agent ──────────────────────────────────────────────────────
#
# Two modes selected automatically by graph state:
#
#   SEND mode  (plan has rag_agent before notifier, and rag_output is set)
#              → visitor asked to have details sent to their email.
#                The agent collects their email, then calls send_profile_email
#                with the pre-fetched rag_output as the body.
#
#   COLLECT mode  (visitor explicitly wants to connect / leave a message)
#              → strict 5-step collection: name → email → note → notify_owner → confirm

def notifier_node(state: TwinState) -> dict:
    """
    Runs the notifier ReAct agent in either SEND or COLLECT mode.

    Guard: if already notified this session, skip to avoid duplicate emails.
    """
    # Duplicate guard
    if state.get("notified", False):
        return {
            "messages": [AIMessage(
                content="✅ I've already passed your message along — I'll follow up with you directly.",
                name="notifier_agent",
            )],
            "knowledge_gap": False,
            "plan_step":     state.get("plan_step", 0) + 1,
            "last_agent":    "notifier_agent",
        }

    # Choose mode based on whether RAG ran before us in this plan
    plan      = state.get("plan", [])
    plan_step = state.get("plan_step", 0)
    rag_ran   = "rag_agent" in plan[:plan_step]
    rag_out   = state.get("rag_output", "")
    send_mode = rag_ran and bool(rag_out) and "KNOWLEDGE_GAP" not in rag_out

    if send_mode:
        # SEND mode — inject rag_output into the prompt so agent knows what to send
        prompt = SystemMessage(content=NOTIFIER_SEND_SYSTEM.content.replace(
            "{{RAG_OUTPUT}}", rag_out
        ))
    else:
        prompt = NOTIFIER_COLLECT_SYSTEM

    tools = [send_email]

    agent = create_react_agent(
        model=_llm(),
        tools=tools,
        prompt=prompt,
    )
    result = agent.invoke(
        {"messages": _human_messages(state)},
        config={"run_name": "notifier-agent"},
    )

    last_ai = result["messages"][-1]
    response_text = _extract_text(
        last_ai.content if isinstance(last_ai, AIMessage) else str(last_ai)
    )

    notified = state.get("notified", False)
    if any(tok in response_text for tok in ("NOTIFICATION_SENT", "NOTIFICATION_LOGGED", "EMAIL_SENT", "EMAIL_LOGGED")):
        notified = True

    return {
        "messages":    [AIMessage(content=response_text, name="notifier_agent")],
        "knowledge_gap": False,
        "notified":    notified,
        "plan_step":   state.get("plan_step", 0) + 1,
        "last_agent":  "notifier_agent",
    }


# ── Node: Chit-Chat Agent ─────────────────────────────────────────────────────

def chitchat_node(state: TwinState) -> dict:
    """Lightweight personality agent — small talk, no tools, minimal tokens."""
    # Use _human_messages so any preceding agent's AIMessage doesn't end the list
    messages = [SystemMessage(content=CHITCHAT_SYSTEM)] + _human_messages(state)
    response = _llm().invoke(
        messages,
        config={"run_name": "chitchat-agent"},
    )
    return {
        "messages":   [AIMessage(content=_extract_text(response.content), name="chitchat_agent")],
        "knowledge_gap": False,
        "plan_step":  state.get("plan_step", 0) + 1,
        "last_agent": "chitchat_agent",
    }


# ── Routing helpers ───────────────────────────────────────────────────────────

_AGENT_NODES = {
    "rag_agent":       "rag_agent",
    "scheduler_agent": "scheduler_agent",
    "notifier_agent":  "notifier_agent",
    "chitchat_agent":  "chitchat_agent",
}


def route_after_supervisor(state: TwinState) -> str:
    """After supervisor sets the plan, jump straight to the first agent."""
    plan = state.get("plan", ["rag_agent"])
    return plan[0] if plan else "rag_agent"


def route_after_agent(state: TwinState) -> str:
    """
    After any agent finishes, decide what to do next:

    1. If RAG detected a knowledge gap → re-plan to notifier (regardless of plan).
    2. If there are more agents in the plan → run the next one.
    3. Otherwise → END.
    """
    # Knowledge gap mid-plan: interrupt and escalate
    if state.get("knowledge_gap", False):
        return "notifier_agent"

    plan      = state.get("plan", [])
    plan_step = state.get("plan_step", 0)

    if plan_step < len(plan):
        next_agent = plan[plan_step]
        return _AGENT_NODES.get(next_agent, END)

    return END


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    """
    Assembles and compiles Sowbhagya's personal AI LangGraph.

    Graph flow:
      START
        → supervisor  (plans a sequence)
        → agent[0]
        → agent[1]    (if plan has 2+ steps)
        → …
        → END

    Any agent can trigger a re-route to notifier_agent via knowledge_gap flag.
    """
    builder = StateGraph(TwinState)

    # Register all nodes
    builder.add_node("supervisor",       supervisor_node)
    builder.add_node("rag_agent",        rag_node)
    builder.add_node("scheduler_agent",  scheduler_node)
    builder.add_node("notifier_agent",   notifier_node)
    builder.add_node("chitchat_agent",   chitchat_node)

    # Entry: always start at supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor → first agent in plan
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "rag_agent":       "rag_agent",
            "scheduler_agent": "scheduler_agent",
            "notifier_agent":  "notifier_agent",
            "chitchat_agent":  "chitchat_agent",
        },
    )

    # After each agent: continue plan or end
    for node in ("rag_agent", "scheduler_agent", "notifier_agent", "chitchat_agent"):
        builder.add_conditional_edges(
            node,
            route_after_agent,
            {
                "rag_agent":       "rag_agent",
                "scheduler_agent": "scheduler_agent",
                "notifier_agent":  "notifier_agent",
                "chitchat_agent":  "chitchat_agent",
                END:               END,
            },
        )

    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)


# ── Convenience: get/cache the compiled graph ─────────────────────────────────

_graph_instance = None


def get_graph():
    """Returns a singleton compiled graph (one per process)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


# ── Initial state factory ─────────────────────────────────────────────────────

def initial_state() -> TwinState:
    return TwinState(
        messages=[],
        plan=[],
        plan_step=0,
        rag_output="",
        knowledge_gap=False,
        visitor_name="",
        visitor_email="",
        booking_state={},
        notified=False,
        last_agent="none",
    )
