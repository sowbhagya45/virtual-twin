"""
Tool implementations for Sowbhagya's personal AI.

Tools:
  retrieve_profile_info  — RAG: semantic search over ChromaDB knowledge base
  create_calendar_event  — Google Calendar: create a real event + send invites to both parties
  send_email             — Gmail SMTP for two scenarios:
                             mode="notify"       → alert Sowbhagya someone wants to connect
                             mode="send_profile" → deliver profile content to visitor's email
"""
from __future__ import annotations

import json
import os
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Lazy vectorstore accessor (loaded once per process) ───────────────────────

_vectorstore = None


def _get_vectorstore():
    """Load ChromaDB vectorstore lazily. Validates the collection is non-empty."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    vs = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="sowbhagya_profile",
    )
    # Validate the collection actually has embeddings — catch silent empty-store failures
    try:
        count = vs._collection.count()
        logger.info(f"ChromaDB loaded: {count} embeddings in 'sowbhagya_profile'")
        if count == 0:
            raise RuntimeError("ChromaDB collection is empty — knowledge base missing")
    except Exception as e:
        logger.error(f"ChromaDB validation failed: {e}")
        raise
    _vectorstore = vs
    return _vectorstore


# ── Google Calendar credentials loader ────────────────────────────────────────

def _get_calendar_service():
    """
    Build and return an authenticated Google Calendar API service object.

    Credential resolution order:
      1. GOOGLE_CALENDAR_TOKEN env var  — JSON string of token.json contents.
         Use this on Streamlit Cloud (paste token.json into Secrets).
      2. token.json file in project root — created by cal_setup.py on first run.

    The credentials.json (OAuth client secret) is only needed during the one-time
    setup flow (cal_setup.py). After that, only token.json is needed.

    Raises RuntimeError if no credentials are found.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    creds = None

    # ── Option 1: token from environment variable (Streamlit Cloud) ──────────
    token_env = os.environ.get("GOOGLE_CALENDAR_TOKEN", "").strip()
    if token_env:
        try:
            token_data = json.loads(token_env)
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        except Exception as e:
            logger.warning("Failed to load GOOGLE_CALENDAR_TOKEN from env: %s", e)

    # ── Option 2: token.json file (local development) ─────────────────────────
    if creds is None:
        token_path = os.path.join(os.path.dirname(__file__), "..", "token.json")
        token_path = os.path.abspath(token_path)
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if creds is None:
        raise RuntimeError(
            "Google Calendar credentials not found.\n"
            "Run `python cal_setup.py` to generate token.json, then set "
            "GOOGLE_CALENDAR_TOKEN in your .env or Streamlit Secrets."
        )

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Persist refreshed token back to file if available
            token_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "token.json")
            )
            if os.path.exists(token_path):
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
        except Exception as e:
            logger.warning("Token refresh failed: %s", e)

    return build("calendar", "v3", credentials=creds)


def _parse_datetime(date_str: str, timezone: str = "Asia/Kolkata") -> str:
    """
    Parse a human date/time string into an RFC 3339 datetime string.

    Handles ISO formats, natural language (e.g. "August 10th at 11:00 AM"),
    and common regional formats.  Always returns a valid RFC 3339 string.
    Falls back to tomorrow 10:00 AM only if everything else fails.
    """
    import re

    s = date_str.strip()

    # ── Step 1: strip ordinal suffixes  "10th" → "10" ────────────────────────
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)

    # ── Step 2: normalise "at" separator  "August 10 at 11:00" → "August 10 11:00"
    s = re.sub(r"\bat\b", " ", s, flags=re.IGNORECASE)

    # ── Step 3: convert 12-hour AM/PM to 24-hour ──────────────────────────────
    def _to24(m):
        h, mi, period = int(m.group(1)), m.group(2), m.group(3).upper()
        if period == "AM":
            h = 0 if h == 12 else h
        else:  # PM
            h = 12 if h == 12 else h + 12
        return f"{h:02d}:{mi}"

    s = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp][Mm])\b", _to24, s)
    # Also handle "11 AM" (no minutes)
    def _to24_short(m):
        h, period = int(m.group(1)), m.group(2).upper()
        if period == "AM":
            h = 0 if h == 12 else h
        else:
            h = 12 if h == 12 else h + 12
        return f"{h:02d}:00"
    s = re.sub(r"\b(\d{1,2})\s*([AaPp][Mm])\b", _to24_short, s)

    # ── Step 4: collapse multiple spaces ─────────────────────────────────────
    s = re.sub(r"\s+", " ", s).strip()

    # ── Step 5: try all format patterns ──────────────────────────────────────
    formats = [
        # ISO / numeric first (most precise)
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        # DD Mon YYYY
        "%d %b %Y %H:%M",
        "%d %B %Y %H:%M",
        "%d %b %Y",
        "%d %B %Y",
        # Mon DD YYYY  (e.g. "August 10 2025 11:00")
        "%B %d %Y %H:%M",
        "%b %d %Y %H:%M",
        "%B %d %Y",
        "%b %d %Y",
        # Mon DD (no year — assume current year, bump to next year if in past)
        "%B %d %H:%M",
        "%b %d %H:%M",
        "%B %d",
        "%b %d",
        # Regional numeric
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
        "%d-%m-%Y %H:%M",
    ]

    current_year = datetime.now().year
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            # If no year in format (strptime defaults to 1900), fill in current year.
            # Only bump to next year if the resulting date is MORE than 1 day in the past
            # — this avoids false bumps for dates like "August 10" when today is Aug 11.
            if dt.year == 1900:
                dt = dt.replace(year=current_year)
                if dt < datetime.now() - timedelta(days=1):
                    dt = dt.replace(year=current_year + 1)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue

    # ── Fallback: tomorrow 10:00 AM ──────────────────────────────────────────
    fallback = (datetime.now() + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    logger.warning("Could not parse datetime '%s', using fallback %s", date_str, fallback)
    return fallback.strftime("%Y-%m-%dT%H:%M:%S")


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


# ── Tool 2: Google Calendar event ─────────────────────────────────────────────

@tool
def create_calendar_event(
    visitor_name: str,
    visitor_email: str,
    purpose: str,
    start_datetime: str,
    duration_minutes: int = 30,
    timezone: str = "Asia/Kolkata",
) -> str:
    """
    Create a real Google Calendar event and send invites to both Sowbhagya and the visitor.

    Call this ONLY after collecting all four required pieces of information:
      visitor_name     — visitor's full name
      visitor_email    — visitor's email address (they will receive a calendar invite)
      purpose          — what the meeting is about (e.g. "Job opportunity at Google")
      start_datetime   — preferred start date and time as a string
                         (e.g. "2025-08-10 11:00", "10th Aug at 11:00 AM", "2025-08-10T11:00:00")
      duration_minutes — meeting length in minutes (default: 30)
      timezone         — IANA timezone string (default: "Asia/Kolkata")
                         Common values: "Asia/Kolkata", "America/New_York", "Europe/London", "UTC"

    What this does:
      - Creates an event on Sowbhagya's Google Calendar
      - Adds the visitor as a guest — Google sends them a calendar invite automatically
      - Sends Sowbhagya an email notification about the booking
      - Returns a BOOKING_CONFIRMED status with the event link

    Returns a status string: BOOKING_CONFIRMED, BOOKING_LOGGED, or BOOKING_FAILED.
    """
    try:
        service = _get_calendar_service()
    except RuntimeError as e:
        # Calendar not configured — fall back to email notification
        logger.warning("Calendar not configured, falling back to email: %s", e)
        return _calendar_email_fallback(visitor_name, visitor_email, purpose, start_datetime)

    start_iso = _parse_datetime(start_datetime, timezone)
    # Compute end time
    try:
        start_dt = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%S")
        end_dt   = start_dt + timedelta(minutes=duration_minutes)
        end_iso  = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        end_iso = start_iso  # fallback: same time (Calendar will handle it)

    owner_email = os.environ.get("OWNER_EMAIL",
                  os.environ.get("GMAIL_USER", "")).strip()

    event_body = {
        "summary": f"Meeting with {visitor_name}",
        "description": (
            f"Meeting requested via Sowbhagya's Personal AI\n\n"
            f"Visitor: {visitor_name} ({visitor_email})\n"
            f"Purpose: {purpose}\n\n"
            f"Booked at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        "start": {
            "dateTime": start_iso,
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_iso,
            "timeZone": timezone,
        },
        "attendees": [
            {"email": visitor_email, "displayName": visitor_name},
        ],
        # Send email invites to all attendees
        "guestsCanSeeOtherGuests": True,
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email",  "minutes": 60},
                {"method": "popup",  "minutes": 10},
            ],
        },
    }

    try:
        event = service.events().insert(
            calendarId="primary",
            body=event_body,
            sendUpdates="all",   # sends Google Calendar invite emails to all attendees
        ).execute()

        event_link = event.get("htmlLink", "")
        event_id   = event.get("id", "")

        logger.info(
            "Calendar event created: %s | %s → %s | link: %s",
            event.get("summary"), start_iso, end_iso, event_link
        )

        return (
            f"BOOKING_CONFIRMED\n\n"
            f"✅ Meeting scheduled!\n"
            f"  With:    {visitor_name} ({visitor_email})\n"
            f"  Purpose: {purpose}\n"
            f"  When:    {start_datetime} ({timezone})\n"
            f"  Duration: {duration_minutes} min\n"
            f"  Event:   {event_link}\n\n"
            f"Both you and {visitor_name} will receive a Google Calendar invite."
        )

    except Exception as exc:
        logger.error("Google Calendar event creation failed: %s", exc)
        # Graceful fallback to email
        return _calendar_email_fallback(visitor_name, visitor_email, purpose, start_datetime)


def _calendar_email_fallback(
    visitor_name: str,
    visitor_email: str,
    purpose: str,
    start_datetime: str,
) -> str:
    """
    Fallback when Google Calendar is unavailable:
    send a plain email to Sowbhagya and return BOOKING_LOGGED.
    """
    gmail_user  = os.environ.get("GMAIL_USER", "").strip()
    app_pwd     = os.environ.get("GMAIL_APP_PWD", "").strip()
    owner_email = os.environ.get("OWNER_EMAIL", gmail_user).strip()
    now         = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    if gmail_user and app_pwd:
        subject = f"[Sowbhagya AI] Meeting request from {visitor_name}"
        body = "\n".join([
            "📅 New Meeting Request (Calendar unavailable — manual booking needed)",
            "",
            f"Name:    {visitor_name}",
            f"Email:   {visitor_email}",
            f"Purpose: {purpose}",
            f"Time:    {start_datetime}",
            "",
            f"Sent at: {now}",
        ])
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"Sowbhagya Mohanthy <{gmail_user}>"
            msg["To"]      = owner_email
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(gmail_user, app_pwd)
                smtp.sendmail(gmail_user, owner_email, msg.as_string())
            logger.info("Fallback booking email sent to %s", owner_email)
        except Exception as e:
            logger.error("Fallback email also failed: %s", e)

    return (
        f"BOOKING_LOGGED\n\n"
        f"Google Calendar is not configured yet. Your request has been recorded:\n"
        f"  {visitor_name} ({visitor_email}) | {purpose} | {start_datetime}\n\n"
        f"Sowbhagya will follow up to confirm the time directly."
    )


# ── Tool 3: Email (notify + send_profile only) ────────────────────────────────
#
# Booking is now handled by create_calendar_event (Tool 2).
# send_email covers the two remaining email-only scenarios:
#   "notify"        → alert Sowbhagya: visitor wants to connect / left a message
#   "send_profile"  → deliver Sowbhagya's profile content to visitor's email
#
# Required env vars:
#   GMAIL_USER     — Sowbhagya's Gmail address
#   GMAIL_APP_PWD  — 16-char Gmail App Password (not login password)
#   OWNER_EMAIL    — where owner-facing alerts go (defaults to GMAIL_USER)

@tool
def send_email(
    mode: str,
    to: str,
    visitor_name: str,
    body: str,
    subject: str = "",
) -> str:
    """
    Send an email for non-booking scenarios.

    Parameters
    ----------
    mode : str
        One of:
          "notify"        — alert Sowbhagya that a visitor wants to connect.
                            `to` = leave empty (auto-fills from OWNER_EMAIL env var).
                            `visitor_name` = visitor's name.
                            `body` = the visitor's message or question.
          "send_profile"  — send Sowbhagya's profile content TO the visitor.
                            `to` = visitor's email address (ask them first).
                            `visitor_name` = visitor's name (for the greeting).
                            `body` = the profile/resume content to deliver.

    to : str
        For "notify": leave empty — OWNER_EMAIL is used automatically.
        For "send_profile": visitor's email address.

    visitor_name : str
        Visitor's name. Used in subject and greeting.

    body : str
        Email content (plain text).

    subject : str, optional
        Overrides the auto-generated subject. Leave empty for default.

    Returns a status string: NOTIFICATION_SENT, EMAIL_SENT, or EMAIL_LOGGED/FAILED.
    """
    gmail_user  = os.environ.get("GMAIL_USER", "").strip()
    app_pwd     = os.environ.get("GMAIL_APP_PWD", "").strip()
    owner_email = os.environ.get("OWNER_EMAIL", gmail_user).strip()
    now         = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    if not gmail_user or not app_pwd:
        logger.warning("GMAIL_USER or GMAIL_APP_PWD not set — email skipped.\n%s", body)
        return (
            f"EMAIL_LOGGED: credentials not configured.\n"
            f"mode={mode} | to={to or owner_email} | visitor={visitor_name}\n"
            f"Body recorded:\n{body}"
        )

    if mode == "notify":
        recipient    = owner_email
        auto_subject = f"[Sowbhagya AI] Message from {visitor_name}"
        full_body    = "\n".join([
            "🔔 Someone wants to connect",
            "",
            f"Visitor: {visitor_name}",
            f"Message: {body}",
            "",
            f"Sent at: {now}",
        ])
        ok_token  = "NOTIFICATION_SENT"
        ok_reply  = f"I'll follow up with {visitor_name} within 24–48 hours."
        err_reply = f"Message recorded for {visitor_name}."

    elif mode == "send_profile":
        if not to:
            return (
                "EMAIL_FAILED: `to` must be the visitor's email address for mode='send_profile'. "
                "Please ask the visitor for their email address first."
            )
        recipient    = to
        auto_subject = "Sowbhagya Mohanthy — Professional Profile"
        full_body    = "\n".join([
            f"Hi {visitor_name},",
            "",
            "Thanks for reaching out! Here are my professional details as requested:",
            "",
            "─" * 60,
            body,
            "─" * 60,
            "",
            "Feel free to reply or connect on LinkedIn:",
            "linkedin.com/in/sowbhagya-mohanthy-8a8a68221/",
            "",
            "Looking forward to connecting!",
            "Sowbhagya Mohanthy",
            "AI Engineer & Software Developer — IBM, Hyderabad",
            "",
            f"Sent at: {now}",
        ])
        ok_token  = "EMAIL_SENT"
        ok_reply  = f"Profile sent to {visitor_name} at {recipient}."
        err_reply = f"Send failed — details recorded for {recipient}."

    else:
        return (
            f"EMAIL_FAILED: unknown mode '{mode}'. "
            "Use 'notify' or 'send_profile'. "
            "For meeting bookings use the create_calendar_event tool instead."
        )

    final_subject = subject if subject else auto_subject

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = final_subject
        msg["From"]    = f"Sowbhagya Mohanthy <{gmail_user}>"
        msg["To"]      = recipient
        msg.attach(MIMEText(full_body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, app_pwd)
            smtp.sendmail(gmail_user, recipient, msg.as_string())

        logger.info("Email sent [%s]: %s → %s", mode, final_subject, recipient)
        return f"{ok_token}\n\n{ok_reply}"

    except smtplib.SMTPAuthenticationError:
        err = (
            "Gmail auth failed. GMAIL_APP_PWD must be a 16-char App Password. "
            "Generate one at myaccount.google.com/apppasswords"
        )
        logger.error(err)
        return f"EMAIL_FAILED: {err}"

    except Exception as exc:
        logger.error("send_email [%s] failed: %s", mode, exc)
        return f"EMAIL_LOGGED: send failed ({exc}). {err_reply}"
