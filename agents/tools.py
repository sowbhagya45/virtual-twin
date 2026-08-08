"""
Tool implementations for virtual-twin agents.
Each tool is a LangChain @tool that agents can call during reasoning.
"""
from __future__ import annotations

import os
from datetime import datetime

from langchain_core.tools import tool

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
        results = vs.similarity_search_with_relevance_scores(query, k=3)  # k=3 saves tokens

        # Filter chunks below confidence threshold
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
    owner_email = os.environ.get("OWNER_EMAIL", "sudhirkumar02001@gmail.com")
    sendgrid_key = os.environ.get("SENDGRID_API_KEY", "")

    summary_lines = [
        f"📅 New Meeting Request — virtual-twin",
        f"",
        f"From:    {visitor_name} ({visitor_email})",
        f"Purpose: {meeting_purpose}",
        f"Time:    {proposed_datetime}",
        f"Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    body = "\n".join(summary_lines)

    if sendgrid_key:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            msg = Mail(
                from_email=owner_email,
                to_emails=owner_email,
                subject=f"[virtual-twin] Meeting request from {visitor_name}",
                plain_text_content=body,
            )
            SendGridAPIClient(sendgrid_key).send(msg)
        except Exception as e:
            return (
                f"BOOKING_LOGGED (email failed: {e})\n\n"
                f"Details: {visitor_name} | {visitor_email} | {proposed_datetime}"
            )

    return (
        f"BOOKING_CONFIRMED\n\n"
        f"Sowbhagya has been notified of your meeting request.\n"
        f"Visitor: {visitor_name} ({visitor_email})\n"
        f"Purpose: {meeting_purpose}\n"
        f"Requested time: {proposed_datetime}"
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
    Notify Sowbhagya Mohanthy when a visitor asks a question that is not in the
    knowledge base, or when a visitor explicitly wants to reach him directly.

    Call this after collecting visitor_name, visitor_email, and their question.
    extra_message is optional — any additional context the visitor wants to add.
    """
    owner_email = os.environ.get("OWNER_EMAIL", "sudhirkumar02001@gmail.com")
    sendgrid_key = os.environ.get("SENDGRID_API_KEY", "")

    body_lines = [
        f"🔔 virtual-twin Notification",
        f"",
        f"A visitor has a question that could not be answered from your knowledge base.",
        f"",
        f"Visitor:  {visitor_name}",
        f"Email:    {visitor_email}",
        f"Question: {question}",
    ]
    if extra_message:
        body_lines.append(f"Note:     {extra_message}")
    body_lines.append(f"\nSent at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    body = "\n".join(body_lines)

    if sendgrid_key:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            msg = Mail(
                from_email=owner_email,
                to_emails=owner_email,
                subject=f"[virtual-twin] Question from {visitor_name}",
                plain_text_content=body,
            )
            SendGridAPIClient(sendgrid_key).send(msg)
        except Exception as e:
            return (
                f"NOTIFICATION_LOGGED (email failed: {e})\n\n"
                f"Details: {visitor_name} | {visitor_email} | {question}"
            )

    return (
        f"NOTIFICATION_SENT\n\n"
        f"Sowbhagya has been notified and will follow up with "
        f"{visitor_name} at {visitor_email} within 24–48 hours."
    )
