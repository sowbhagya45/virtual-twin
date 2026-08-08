"""
Tool implementations for Sowbhagya's personal AI.
Each tool is a LangChain @tool that agents can call during reasoning.
"""
from __future__ import annotations

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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


# ── Email helper — Gmail SMTP (free, no third-party service) ──────────────────

def _send_email(subject: str, body: str) -> tuple[bool, str]:
    """
    Send an email notification via Gmail SMTP using Python's built-in smtplib.

    Completely free — no SendGrid, no third-party service, no sender verification.
    Sends directly from your Gmail account using an App Password.

    Required env vars:
      GMAIL_USER     — your Gmail address  (e.g. sudhirkumar02001@gmail.com)
      GMAIL_APP_PWD  — 16-char Gmail App Password (NOT your Gmail login password)

    Optional:
      OWNER_EMAIL    — delivery address for notifications (defaults to GMAIL_USER)

    One-time Gmail setup (2 minutes):
      1. Enable 2-Step Verification: myaccount.google.com/security
      2. Generate App Password:      myaccount.google.com/apppasswords
         → Select app: "Mail"  → Select device: "Other" → name it "personal-ai"
         → Copy the 16-char password into GMAIL_APP_PWD (spaces don't matter)

    Returns (success: bool, status_message: str).
    """
    gmail_user  = os.environ.get("GMAIL_USER", "").strip()
    app_pwd     = os.environ.get("GMAIL_APP_PWD", "").strip()
    owner_email = os.environ.get("OWNER_EMAIL", gmail_user).strip()

    if not gmail_user or not app_pwd:
        logger.warning(
            "GMAIL_USER or GMAIL_APP_PWD not configured — email skipped.\n%s", body
        )
        return False, "EMAIL_SKIPPED: set GMAIL_USER and GMAIL_APP_PWD in .env"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"Sowbhagya AI <{gmail_user}>"
        msg["To"]      = owner_email
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, app_pwd)
            smtp.sendmail(gmail_user, owner_email, msg.as_string())

        logger.info("Email sent: %s → %s", subject, owner_email)
        return True, "EMAIL_SENT"

    except smtplib.SMTPAuthenticationError:
        err = (
            "Gmail authentication failed. "
            "GMAIL_APP_PWD must be a 16-char App Password, not your Gmail login. "
            "Generate one at myaccount.google.com/apppasswords"
        )
        logger.error(err)
        return False, f"EMAIL_FAILED: {err}"

    except Exception as exc:
        err = str(exc)
        logger.error("Email send exception: %s", err)
        return False, f"EMAIL_FAILED: {err}"


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
            sections.append(
                f"[source: {source} | relevance: {score:.2f}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(sections)

    except Exception as e:
        return f"KNOWLEDGE_GAP: Could not retrieve from knowledge base — {e}"


# ── Tool 2: Meeting booking ────────────────────────────────────────────────────

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
    - visitor_email: email address of the visitor
    - meeting_purpose: reason for the meeting (job opportunity, collaboration, etc.)
    - proposed_datetime: visitor's preferred date/time string

    Sends a notification email to Sowbhagya via Gmail SMTP.
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
        return (
            f"BOOKING_LOGGED ({status_msg})\n\n"
            f"Your request has been recorded: {visitor_name} | {visitor_email} | {proposed_datetime}\n"
            f"I'll follow up with you directly."
        )


# ── Tool 3: Notify owner ───────────────────────────────────────────────────────

@tool
def notify_owner(
    visitor_name: str,
    visitor_email: str,
    question: str,
    extra_message: str = "",
) -> str:
    """
    Notify Sowbhagya Mohanthy when a visitor wants to reach him directly or asks
    something outside the knowledge base.

    Call this after collecting visitor_name, visitor_email, and their question.
    extra_message is optional — any additional context the visitor wants to add.

    Sends a notification email to Sowbhagya via Gmail SMTP.
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
            f"Your message has been recorded. I'll follow up with {visitor_name} directly."
        )
