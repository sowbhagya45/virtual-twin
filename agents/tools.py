"""
Tool implementations for Sowbhagya's personal AI.
Each tool is a LangChain @tool that agents can call during reasoning.
"""
from __future__ import annotations

import os
import logging
from datetime import datetime

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Lazy vectorstore accessor (loaded once per process) ───────────────────────
_vectorstore = None


def _get_vectorstore():
    """Load ChromaDB vectorstore lazily so it's only built after ingest.py runs."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    _vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="sowbhagya_profile",
    )
    return _vectorstore


def _send_email(subject: str, body: str) -> tuple[bool, str]:
    """
    Send an email via SendGrid.

    Returns (success: bool, message: str).

    SendGrid requirement: the from_email address MUST be a verified sender
    in your SendGrid account (Single Sender Verification or Domain Auth).

    Env vars:
      SENDGRID_API_KEY   — SendGrid API key (required to send)
      OWNER_EMAIL        — recipient (your inbox, e.g. sudhirkumar02001@gmail.com)
      SENDGRID_FROM_EMAIL — verified sender address in SendGrid
                            Defaults to OWNER_EMAIL if not set.
                            If OWNER_EMAIL isn't verified in SendGrid, set this
                            to a dedicated verified address like noreply@yourdomain.com
    """
    sendgrid_key = os.environ.get("SENDGRID_API_KEY", "").strip()
    owner_email  = os.environ.get("OWNER_EMAIL", "sudhirkumar02001@gmail.com").strip()
    # from_email must be verified in SendGrid — defaults to owner_email but
    # can be overridden by SENDGRID_FROM_EMAIL for a dedicated verified sender
    from_email   = os.environ.get("SENDGRID_FROM_EMAIL", owner_email).strip()

    if not sendgrid_key:
        # No SendGrid key — log locally so it's visible in terminal / Streamlit logs
        logger.warning("SENDGRID_API_KEY not set — email not sent. Body:\n%s", body)
        return False, "EMAIL_SKIPPED: no SENDGRID_API_KEY configured"

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        msg = Mail(
            from_email=from_email,
            to_emails=owner_email,
            subject=subject,
            plain_text_content=body,
        )
        resp = SendGridAPIClient(sendgrid_key).send(msg)
        status = resp.status_code

        if status in (200, 202):
            return True, f"EMAIL_SENT (HTTP {status})"
        else:
            # Non-success HTTP — surface the status code
            err = f"SendGrid returned HTTP {status}"
            logger.error("Email send failed: %s", err)
            return False, f"EMAIL_FAILED: {err}"

    except Exception as exc:
        # Capture the full error message — common causes:
        #   403 → from_email not verified in SendGrid
        #   401 → invalid API key
        err_msg = str(exc)
        logger.error("Email send exception: %s", err_msg)
        return False, f"EMAIL_FAILED: {err_msg}"


# ── Tool 1: RAG retrieval ──────────────────────────────────────────────────────

@tool
def retrieve_profile_info(query: str) -> str:
    """
    Retrieve relevant information about Sowbhagya Mohanthy from his professional
    knowledge base (resume, projects, skills, experience, education, certifications).

    Use this tool for ANY question about:
    - Skills (AI, ML, LangGraph, LangChain, Python, FastAPI, React, etc.)
    - Work experience at IBM (projects, roles, achievements)
    - Education background
    - Certifications
    - Contact details or social profiles
    - Availability or employment status

    Returns the most relevant chunks from the knowledge base, or a KNOWLEDGE_GAP
    signal if nothing relevant is found.
    """
    try:
        vs = _get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(query, k=3)

        relevant = [
            (doc, score)
            for doc, score in results
            if score >= 0.35
        ]

        if not relevant:
            return (
                "KNOWLEDGE_GAP: No relevant information found in the knowledge base "
                f"for the query: '{query}'. Please escalate to the notifier agent."
            )

        sections = []
        for doc, score in relevant:
            source = doc.metadata.get("source", "resume")
            sections.append(f"[source: {source} | relevance: {score:.2f}]\n{doc.page_content}")

        return "\n\n---\n\n".join(sections)

    except Exception as e:
        return f"KNOWLEDGE_GAP: Could not retrieve from knowledge base — {e}"


# ── Tool 2: Calendar / meeting booking ────────────────────────────────────────

@tool
def book_meeting(
    visitor_name: str,
    visitor_email: str,
    meeting_purpose: str,
    proposed_datetime: str,
) -> str:
    """
    Confirm a meeting booking request with Sowbhagya Mohanthy.

    Call this after collecting:
    - visitor_name: full name of the visitor
    - visitor_email: email address to send the calendar invite to
    - meeting_purpose: reason for the meeting (job opportunity, collaboration, etc.)
    - proposed_datetime: visitor's preferred date/time string

    This sends a notification email to Sowbhagya with the meeting request details.
    """
    subject = f"[Sowbhagya AI] Meeting request from {visitor_name}"
    body = "\n".join([
        "📅 New Meeting Request",
        "",
        f"From:    {visitor_name} ({visitor_email})",
        f"Purpose: {meeting_purpose}",
        f"Time:    {proposed_datetime}",
        f"Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
    ])

    ok, status_msg = _send_email(subject, body)

    if ok:
        return (
            f"BOOKING_CONFIRMED\n\n"
            f"I've been notified of your meeting request.\n"
            f"Visitor: {visitor_name} ({visitor_email})\n"
            f"Purpose: {meeting_purpose}\n"
            f"Requested time: {proposed_datetime}"
        )
    else:
        # Still acknowledge the booking even if email failed —
        # the status_msg surfaces the real error in the LangSmith trace
        return (
            f"BOOKING_LOGGED ({status_msg})\n\n"
            f"Your request has been recorded: {visitor_name} | {visitor_email} | {proposed_datetime}\n"
            f"Note: email notification encountered an issue — Sowbhagya will follow up directly."
        )


# ── Tool 3: Notify owner of unknown question / direct contact request ─────────

@tool
def notify_owner(
    visitor_name: str,
    visitor_email: str,
    question: str,
    extra_message: str = "",
) -> str:
    """
    Notify Sowbhagya Mohanthy when a visitor asks a question not in the knowledge
    base, or when a visitor explicitly wants to reach him directly.

    Call this after collecting visitor_name, visitor_email, and their question.
    extra_message is optional — any additional context the visitor wants to add.
    """
    subject = f"[Sowbhagya AI] Message from {visitor_name}"
    body_lines = [
        "🔔 Someone wants to connect",
        "",
        f"Visitor:  {visitor_name}",
        f"Email:    {visitor_email}",
        f"Message:  {question}",
    ]
    if extra_message:
        body_lines.append(f"Note:     {extra_message}")
    body_lines.append(f"\nSent at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    body = "\n".join(body_lines)

    ok, status_msg = _send_email(subject, body)

    if ok:
        return (
            f"NOTIFICATION_SENT\n\n"
            f"I'll follow up with {visitor_name} at {visitor_email} within 24–48 hours."
        )
    else:
        return (
            f"NOTIFICATION_LOGGED ({status_msg})\n\n"
            f"Your message has been recorded: {visitor_name} | {visitor_email}\n"
            f"Note: email notification encountered an issue — I'll still follow up directly."
        )
