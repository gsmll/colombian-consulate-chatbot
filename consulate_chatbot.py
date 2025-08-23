import os
import logging
import re
import json
import hashlib
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from openai import OpenAI
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request, jsonify, Response
from pathlib import Path
from PyPDF2 import PdfReader

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration container with validation"""
    OPENAI_API_KEY: str
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    OPENAI_ORG_ID: Optional[str] = None
    OPENAI_PROJECT_ID: Optional[str] = None
    OPENAI_ASSISTANT_ID: Optional[str] = None
    GOOGLE_CALENDAR_ID: Optional[str] = None
    GOOGLE_SERVICE_ACCOUNT_FILE: Optional[str] = None
    TIMEZONE: str = "America/Chicago"
    APPOINTMENT_DURATION_MINUTES: int = 30

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        required_vars = [
            'OPENAI_API_KEY',
            'TWILIO_ACCOUNT_SID',
            'TWILIO_AUTH_TOKEN'
        ]
        
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
            
        return cls(
            OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY'),
            TWILIO_ACCOUNT_SID=os.environ.get('TWILIO_ACCOUNT_SID'),
            TWILIO_AUTH_TOKEN=os.environ.get('TWILIO_AUTH_TOKEN'),
            OPENAI_ORG_ID=os.environ.get('OPENAI_ORG_ID'),
            OPENAI_PROJECT_ID=os.environ.get('OPENAI_PROJECT_ID'),
            OPENAI_ASSISTANT_ID=os.environ.get('OPENAI_ASSISTANT_ID'),
            GOOGLE_CALENDAR_ID=os.environ.get('GOOGLE_CALENDAR_ID'),
            GOOGLE_SERVICE_ACCOUNT_FILE=(
                os.environ.get('GOOGLE_SERVICE_ACCOUNT_FILE')
                or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            ),
            TIMEZONE=os.environ.get('TIMEZONE', 'America/Chicago'),
            APPOINTMENT_DURATION_MINUTES=int(os.environ.get('APPOINTMENT_DURATION_MINUTES', '30')),
        )


# Removed IntentDetector; single-model interpretation will be used


class AppointmentManager:
    """Manage appointments via Google Calendar with per-user monthly cap."""

    def __init__(self, config: Config):
        self.cfg = config
        self.tz = ZoneInfo(self.cfg.TIMEZONE)
        self.scopes = ['https://www.googleapis.com/auth/calendar']
        self.calendar_id = self.cfg.GOOGLE_CALENDAR_ID or 'primary'
        self.duration = timedelta(minutes=self.cfg.APPOINTMENT_DURATION_MINUTES)
        self.service = self._build_calendar_service()
        self._verify_calendar_access()
        logger.info(
            f"Calendar config: id={self.calendar_id}, tz={self.cfg.TIMEZONE}, duration_min={self.cfg.APPOINTMENT_DURATION_MINUTES}"
        )

    def _build_calendar_service(self):
        creds = None
        try:
            sa_path = self.cfg.GOOGLE_SERVICE_ACCOUNT_FILE
            if not sa_path:
                raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_APPLICATION_CREDENTIALS env var.")
            logger.info(f"Using Google credentials file at: {os.path.abspath(sa_path)}")
            creds = service_account.Credentials.from_service_account_file(sa_path, scopes=self.scopes)
            # Log the service account email to aid configuration
            try:
                sa_email = getattr(creds, 'service_account_email', None)
                if sa_email:
                    logger.info(f"Google service account: {sa_email}")
            except Exception:
                pass
            http = httplib2.Http(timeout=10)
            return build('calendar', 'v3', credentials=creds, cache_discovery=False, http=http)
        except Exception as e:
            logger.error(f"Google Calendar auth/build error: {e}")
            raise

    def _verify_calendar_access(self) -> None:
        """Fail fast if the calendar ID is invalid or not shared with the service account."""
        try:
            info = self.service.calendars().get(calendarId=self.calendar_id).execute(num_retries=2)
            logger.info(f"Google Calendar ready: {info.get('summary')} ({self.calendar_id})")
        except HttpError as he:
            status = getattr(he.resp, 'status', None)
            if status in (403, 404):
                logger.error(
                    "Calendar not found or no access. Ensure GOOGLE_CALENDAR_ID is correct and the calendar is shared with the service account (Make changes to events)."
                )
            raise

    # Utility time helpers
    def _now(self) -> datetime:
        return datetime.now(self.tz)

    def _month_window(self, ref: Optional[datetime] = None) -> tuple[datetime, datetime]:
        ref = ref or self._now()
        start = ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # next month start
        if start.month == 12:
            next_start = start.replace(year=start.year + 1, month=1)
        else:
            next_start = start.replace(month=start.month + 1)
        return start, next_start

    def _format_rfc3339(self, dt: datetime) -> str:
        # Ensure timezone aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.isoformat()

    # Search helpers
    def _list_events(self, time_min: datetime, time_max: datetime, q: Optional[str] = None, 
                     filter_func: Optional[Callable[[dict], bool]] = None) -> List[dict]:
        events: List[dict] = []
        page_token = None
        while True:
            req = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=self._format_rfc3339(time_min),
                timeMax=self._format_rfc3339(time_max),
                singleEvents=True,
                orderBy='startTime',
                q=q,
                pageToken=page_token,
            )
            res = req.execute(num_retries=2)
            batch = res.get('items', [])
            if filter_func:
                batch = [e for e in batch if filter_func(e)]
            events.extend(batch)
            page_token = res.get('nextPageToken')
            if not page_token:
                break
        return events

    def _user_events_in_month(self, user_key: str) -> List[dict]:
        start, end = self._month_window()
        return self._list_events(
            start, end, q=None,
            filter_func=lambda e: (
                e.get('extendedProperties', {})
                 .get('private', {})
                 .get('user_key') == user_key
            )
        )

    def _upcoming_user_events(self, user_key: str) -> List[dict]:
        now = self._now()
        return [
            e for e in self._list_events(
                now - timedelta(days=1), now + timedelta(days=365), q=None,
                filter_func=lambda e: (
                    e.get('status') != 'cancelled' and (
                        e.get('extendedProperties', {})
                         .get('private', {})
                         .get('user_key') == user_key
                    )
                )
            )
        ]

    def _has_conflict(self, start: datetime, end: datetime) -> bool:
        # Check for overlapping events in the target window
        events = self._list_events(start - timedelta(minutes=1), end + timedelta(minutes=1))
        for e in events:
            e_start_str = e.get('start', {}).get('dateTime')
            e_end_str = e.get('end', {}).get('dateTime')
            if not e_start_str or not e_end_str:
                continue
            e_start = datetime.fromisoformat(e_start_str.replace('Z', '+00:00')).astimezone(self.tz)
            e_end = datetime.fromisoformat(e_end_str.replace('Z', '+00:00')).astimezone(self.tz)
            latest_start = max(start, e_start)
            earliest_end = min(end, e_end)
            if latest_start < earliest_end:
                return True
        return False

    def _next_business_slot(self, start_from: Optional[datetime] = None) -> datetime:
        cur = (start_from or self._now())
        # Round up to next 30-min slot
        minute = (cur.minute // 30 + 1) * 30
        if minute == 60:
            cur = cur.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            cur = cur.replace(minute=minute, second=0, microsecond=0)
        # Business hours 08:00-13:00 (Atención al público)
        while True:
            # Skip weekends
            if cur.weekday() >= 5:
                # move to next Monday 09:00
                days_ahead = 7 - cur.weekday()
                cur = cur + timedelta(days=days_ahead)
                cur = cur.replace(hour=8, minute=0, second=0, microsecond=0)
                continue
            if cur.hour < 8:
                cur = cur.replace(hour=8, minute=0, second=0, microsecond=0)
            if cur.hour >= 13:
                # move to next day 08:00
                cur = (cur + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
                continue
            # Check conflict
            end = cur + self.duration
            if not self._has_conflict(cur, end):
                return cur
            # Move to next slot
            cur = cur + timedelta(minutes=30)

    def next_n_slots(self, n: int = 5, start_from: Optional[datetime] = None, day_filter: Optional[int] = None) -> List[datetime]:
        """Return next n available start times. Optional day_filter is weekday index 0-6."""
        slots: List[datetime] = []
        cur = self._next_business_slot(start_from)
        guard = 0
        while len(slots) < n and guard < 2000:
            guard += 1
            if day_filter is not None and cur.weekday() != day_filter:
                # jump to next day at 08:00
                cur = (cur + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
                cur = self._next_business_slot(cur)
                continue
            if self._validate_business_time(cur) and not self._has_conflict(cur, cur + self.duration):
                slots.append(cur)
            # advance 30 minutes
            cur = cur + timedelta(minutes=30)
            # roll over end of day to next day start
            if cur.hour >= 13:
                cur = (cur + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
                cur = self._next_business_slot(cur)
        return slots

    def _validate_business_time(self, when: datetime) -> bool:
        if when.weekday() >= 5:
            return False
        local = when
        if local.hour < 8 or local.hour >= 13:
            return False
        return True

    def book_at(self, user_key: str, when: datetime) -> Dict[str, Any]:
        # Enforce monthly limit; if over limit, return a specific code so the bot can prompt user
        monthly = self._user_events_in_month(user_key)
        if len(monthly) >= 3:
            return {"success": False, "code": "MONTHLY_CAP", "message": "Has alcanzado el límite de 3 citas este mes."}
        if not self._validate_business_time(when):
            return {"success": False, "message": "El horario de atención es de lunes a viernes de 8:00 a.m. a 1:00 p.m."}
        # Round to nearest 30-min slot
        minute = (when.minute // 30) * 30
        when = when.replace(minute=minute, second=0, microsecond=0)
        end = when + self.duration
        if self._has_conflict(when, end):
            return {"success": False, "message": "Ese horario no está disponible. ¿Deseas el siguiente disponible?"}
        created = self.service.events().insert(
            calendarId=self.calendar_id,
            body=self._event_payload(user_key, when)
        ).execute(num_retries=2)
        return {
            "success": True,
            "message": f"Cita confirmada para {when.strftime('%d/%m/%Y %H:%M')}. ID: {created.get('id')}",
            "event": created,
        }

    def _event_payload(self, user_key: str, when: datetime, summary: str = "Consulate Appointment") -> dict:
        end = when + self.duration
        return {
            'summary': summary,
            'description': f'User: {user_key}\nOrigin: Chatbot',
            'start': {
                'dateTime': self._format_rfc3339(when),
                'timeZone': self.cfg.TIMEZONE,
            },
            'end': {
                'dateTime': self._format_rfc3339(end),
                'timeZone': self.cfg.TIMEZONE,
            },
            'extendedProperties': {
                'private': {
                    'user_key': user_key
                }
            }
        }

    def book(self, user_key: str) -> Dict[str, Any]:
        # Enforce monthly limit; return code so bot can prompt to cancel
        monthly = self._user_events_in_month(user_key)
        if len(monthly) >= 3:
            return {"success": False, "code": "MONTHLY_CAP", "message": "Has alcanzado el límite de 3 citas este mes."}
        # Find next available slot and create event
        when = self._next_business_slot()
        body = self._event_payload(user_key, when)
        created = self.service.events().insert(calendarId=self.calendar_id, body=body).execute(num_retries=2)
        start_dt = when.strftime('%d/%m/%Y %H:%M')
        return {
            "success": True,
            "message": f"Cita confirmada para {start_dt}. ID: {created.get('id')}",
            "event": created,
        }

    def cancel_next(self, user_key: str) -> Dict[str, Any]:
        # Find next upcoming event for this phone and delete it
        now = self._now()
        upcoming = [
            e for e in self._list_events(
                now - timedelta(days=1), now + timedelta(days=365), q=None,
                filter_func=lambda e: (
                    e.get('status') != 'cancelled' and (
                        e.get('extendedProperties', {})
                         .get('private', {})
                         .get('user_key') == user_key
                    )
                )
            )
        ]
        if not upcoming:
            return {"success": False, "message": "No tienes ninguna cita programada."}
        # pick the earliest upcoming
        def start_of(e):
            s = e.get('start', {}).get('dateTime')
            return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(self.tz)
        upcoming.sort(key=start_of)
        target = upcoming[0]
        self.service.events().delete(calendarId=self.calendar_id, eventId=target['id']).execute(num_retries=2)
        return {"success": True, "message": f"Cita {target['id']} cancelada exitosamente."}

    def cancel_all(self, user_key: str) -> Dict[str, Any]:
        """Cancel all upcoming appointments for this user."""
        upcoming = self._upcoming_user_events(user_key)
        if not upcoming:
            return {"success": True, "message": "No tienes citas futuras que cancelar."}
        count = 0
        for e in upcoming:
            try:
                self.service.events().delete(calendarId=self.calendar_id, eventId=e['id']).execute(num_retries=2)
                count += 1
            except Exception:
                pass
        return {"success": True, "message": f"Se cancelaron {count} cita(s)."}

    def check_next(self, user_key: str) -> Dict[str, Any]:
        now = self._now()
        upcoming = [
            e for e in self._list_events(
                now - timedelta(days=1), now + timedelta(days=365), q=None,
                filter_func=lambda e: (
                    e.get('status') != 'cancelled' and (
                        e.get('extendedProperties', {})
                         .get('private', {})
                         .get('user_key') == user_key
                    )
                )
            )
        ]
        if not upcoming:
            return {"success": False, "message": "No tienes ninguna cita programada."}
        def start_of(e):
            s = e.get('start', {}).get('dateTime')
            return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(self.tz)
        upcoming.sort(key=start_of)
        first = upcoming[0]
        when = start_of(first)
        return {
            "success": True,
            "message": f"Tu próxima cita es el {when.strftime('%d/%m/%Y a las %H:%M')}. ID: {first['id']}"
        }

    def list_upcoming(self, user_key: str) -> List[dict]:
        """Return all upcoming events sorted by start time."""
        events = self._upcoming_user_events(user_key)
        def start_of(e):
            s = e.get('start', {}).get('dateTime')
            return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(self.tz)
        events.sort(key=start_of)
        return events

class ConsulateBot:
    """Main chatbot class-- handles OpenAI, Twilio, and appointments"""
    
    def __init__(self, config: Config):
        self.config = config
        # Log masked API key presence to help diagnose auth issues
        try:
            ak = (config.OPENAI_API_KEY or "").strip()
            masked = (ak[:4] + "…" + ak[-4:]) if ak else "(missing)"
            logger.info(f"OpenAI API key: {masked}")
        except Exception:
            pass
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=config.OPENAI_API_KEY
        )
        # Optional request timeout for OpenAI calls (seconds)
        try:
            self.openai_timeout = int(os.environ.get('OPENAI_TIMEOUT', '12'))
        except Exception:
            self.openai_timeout = 12
        self.threads = {}  # In-memory message history per user id
        self.max_history = 2  # messages to keep per user
        self.twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    # No separate intent detector; we use a single-model interpreter per message
        # Short-lived pending actions per user (e.g., cap cancel -> then proceed)
        self.pending_actions: Dict[str, Dict[str, Any]] = {}
        # Quick auth check; only disable client on real auth errors
        try:
            _ = self.openai_client.models.list()
            logger.info("OpenAI auth OK")
        except Exception as e:
            msg = str(e)
            logger.error(f"OpenAI auth check failed: {msg}")
            auth_error = ("Missing bearer" in msg) or ("401" in msg) or ("invalid_api_key" in msg)
            if auth_error:
                self.openai_client = None
                # Interpreter will be disabled implicitly when openai_client is None
            else:
                logger.warning("Proceeding with OpenAI client despite healthcheck error (non-auth).")
        try:
            self.appointments = AppointmentManager(config)
        except Exception as e:
            logger.warning(f"Appointment manager disabled due to error: {e}")
            self.appointments = None
        # Load PDF once for grounding
        try:
            self.pdf_path = Path(__file__).parent / 'consulate_information.pdf'
            self.pdf_corpus = self._load_pdf_text(self.pdf_path) if self.pdf_path.exists() else ""
            if self.pdf_corpus:
                logger.info(f"Loaded consulate_information.pdf for grounding: {len(self.pdf_corpus)} chars")
            else:
                logger.warning("PDF file not found or empty")
        except Exception as e:
            logger.warning(f"Failed to load PDF for grounding: {e}")

    def _route_or_answer(self, user_msg: str, history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Single LLM call that decides between appointment intent vs. general answer.
        Returns one of:
          {"mode":"appointment", "action":"book|cancel|check|availability|cancel_all|list", "when_iso"?: str}
          {"mode":"answer", "text": str}
        """
        if not self.openai_client:
            return None
        grounding = self._retrieve_context(user_msg)
        system_router = (
            "You are an assistant for a consulate. Decide if the user message is about appointments, or a general consular question. "
            "If it is about appointments, output JSON with mode=appointment and an action in {book,cancel,check,availability,cancel_all,list}. "
            "If a specific date/time is mentioned, include when_iso in ISO 8601 (YYYY-MM-DDTHH:MM) without timezone. "
            "If it is a general question, answer concisely (1-2 sentences) in the user's language and output JSON with mode=answer and text. "
            "Do not include any keys other than those requested. Output only JSON."
        )
        system_context = (
            f"Use ONLY this context to answer general questions. If insufficient, say you don't know.\n{grounding}"
            if grounding else "If no context is provided, say you don't know."
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_router},
            {"role": "system", "content": system_context},
            *history[-self.max_history:],
            {"role": "user", "content": user_msg},
        ]
        try:
            try:
                comp = self.openai_client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=messages,
                    timeout=self.openai_timeout,
                )
            except TypeError:
                # SDK may not support timeout kwarg
                comp = self.openai_client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=messages,
                )
            raw = self._extract_text_reply(comp)
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("mode") in {"appointment", "answer"}:
                return data
        except Exception as e:
            logger.warning(f"Router call failed: {e}")
        return None

    def _load_pdf_text(self, path: Path) -> str:
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        # Light cleanup
        return "\n".join(t.strip() for t in texts if t.strip())[:120_000]

    # Paragraph splitting not needed with full-PDF grounding

    def _retrieve_context(self, query: str, k: int = 8) -> str:
        """Small PDF path: always return the full corpus (capped) so answers work if the PDF changes."""
        try:
            total_chars = len(self.pdf_corpus or "")
        except Exception:
            total_chars = 0
        if total_chars == 0:
            logger.warning("No PDF corpus loaded")
            return ""
        logger.info("Using full PDF corpus as context")
        return (self.pdf_corpus or "")[:6000]

    # Removed deterministic extractors; always answer from PDF context with the model

    def _normalize_phone(self, raw: str) -> str:
        """Extract digits so WhatsApp/SMS prefixes don't fragment identity."""
        digits = ''.join(ch for ch in (raw or '') if ch.isdigit())
        if digits:
            return digits
        # Fall back to a short stable hash if we don't have digits (e.g., some channel IDs)
        h = hashlib.sha256((raw or '').encode('utf-8')).hexdigest()
        return h[:24]

    def _extract_text_reply(self, comp: Any) -> str:
        """Extract plain text from a chat.completions response safely."""
        try:
            if not comp or not getattr(comp, "choices", None):
                return ""
            choice = comp.choices[0]
            # Log finish reason and usage if available
            try:
                fr = getattr(choice, "finish_reason", None)
                usage = getattr(comp, "usage", None)
                logger.info(f"LLM finish_reason={fr} usage={usage}")
            except Exception:
                pass
            msg = getattr(choice, "message", None)
            if not msg:
                return ""
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content.strip()
            # Some SDKs may return list of parts
            if isinstance(content, list):
                text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                return "\n".join(t for t in text_parts if t).strip()
            return ""
        except Exception:
            return ""

    def _interpret_appointment_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """Use the LLM to parse free-form user text into a structured appointment command.
        Returns dict like { action: 'book'|'cancel'|'check'|'availability'|'cancel_all', when_iso?: str }.
        """
        if not self.openai_client:
            return None
        try:
            comp = self.openai_client.chat.completions.create(
                model="gpt-5-nano",
        messages=[
                    {"role": "system", "content": (
                        "You convert user messages about appointments into a strict JSON. "
            "Allowed actions: book, cancel, check, availability, cancel_all, list. "
                        "If a date/time is mentioned, include when_iso in ISO 8601 (YYYY-MM-DDTHH:MM) without timezone. "
                        "If the user wants to cancel a specific appointment, set action=cancel and when_iso. "
                        "If they want to cancel all appointments, action=cancel_all. "
                        "Language can be English or Spanish. Output only JSON."
                    )},
                    {"role": "user", "content": text[:500]},
                ]
            )
            raw = self._extract_text_reply(comp)
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _parse_requested_time(self, text: str) -> Optional[datetime]:
        """Parse simple requests like 'monday at 1', '8/25 at 1:00 p', 'lunes 1pm', 'today 10', 'tomorrow at noon'."""
        t = (text or "").lower()
        now = datetime.now(ZoneInfo(self.config.TIMEZONE))
        # Try explicit date like 8/25 or 08/25/2025
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", t)
        hour = None
        minute = 0
        ampm = None
        # 'noon' and 'mediodia'
        if "noon" in t or "medio" in t:
            hour = 12
            minute = 0
            ampm = "pm"
        if "midnight" in t:
            hour = 0
            minute = 0
            ampm = "am"
        hm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", t)
        if hm:
            hour = int(hm.group(1))
            minute = int(hm.group(2) or 0)
            ampm = (hm.group(3) or "").replace(".", "")
        # Weekday names
        weekdays_en = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        weekdays_es = ["lunes","martes","miercoles","miércoles","jueves","viernes","sabado","sábado","domingo"]
        wd_idx = None
        # today/tomorrow
        if any(w in t for w in ["today","hoy"]):
            wd_idx = now.weekday()
        elif any(w in t for w in ["tomorrow","mañana"]):
            wd_idx = (now.weekday() + 1) % 7
        for i, name in enumerate(weekdays_en):
            if name in t:
                wd_idx = i
                break
        if wd_idx is None:
            for i, name in enumerate(weekdays_es):
                if name in t:
                    # map spanish to python weekday index
                    mapping = {"lunes":0,"martes":1,"miercoles":2,"miércoles":2,"jueves":3,"viernes":4,"sabado":5,"sábado":5,"domingo":6}
                    wd_idx = mapping[name]
                    break
        # Compute base date
        dt_date = None
        if m:
            month = int(m.group(1))
            day = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else now.year
            if year < 100:
                year += 2000
            try:
                dt_date = now.replace(year=year, month=month, day=day)
            except ValueError:
                return None
        elif wd_idx is not None:
            # next occurrence of that weekday (including today if later)
            days_ahead = (wd_idx - now.weekday()) % 7
            if days_ahead == 0 and hour is not None and (now.hour > hour):
                days_ahead = 7
            dt_date = (now + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return None
        # Time component
        if hour is None:
            hour = 9  # default morning hour if not provided
        if ampm:
            if ampm.startswith('p') and hour < 12:
                hour += 12
            if ampm.startswith('a') and hour == 12:
                hour = 0
        return dt_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _fallback_from_context(self, query: str, context: str) -> str:
        """Pick 1–2 relevant sentences from the context as a last-resort answer."""
        if not context:
            return ""
        q = (query or "").lower()
        # Split into sentences conservatively
        parts = re.split(r"(?<=[\.!?])\s+|\n+|•|\u2022", context)
        parts = [p.strip() for p in parts if p and len(p.strip()) > 5]
        if not parts:
            return ""
        # Prefer sentences that include any query tokens
        words = [w for w in re.findall(r"\w+", q) if len(w) > 3]
        def score(sent: str) -> int:
            s = sent.lower()
            return sum(1 for w in set(words) if w in s)
        parts.sort(key=score, reverse=True)
        snippet = ". ".join(parts[:2])
        return snippet[:500]

    # Removed fee extractor helper

    def get_or_create_thread(self, user_id: str) -> List[Dict[str, str]]:
        """Get existing chat history list or create new one for the user."""
        if user_id not in self.threads:
            self.threads[user_id] = []
        return self.threads[user_id]

    def process_message(self, phone_number: str, incoming_msg: str) -> str:
        """Process incoming message and return response, with appointment intents."""
        if incoming_msg.lower() == "quit":
            if phone_number in self.threads:
                del self.threads[phone_number]
            return "Exiting the chat. Goodbye!"
        try:
            user_key = self._normalize_phone(phone_number)
            # Handle pending ephemeral flows (e.g., monthly-cap cancel selection)
            pend = self.pending_actions.get(user_key)
            # Enforce TTL (15 minutes) for pending actions
            if pend and isinstance(pend, dict):
                try:
                    ts_str = pend.get("ts")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str)
                        now_utc = datetime.now(timezone.utc)
                        if (now_utc - ts) > timedelta(minutes=15):
                            self.pending_actions.pop(user_key, None)
                            pend = None
                except Exception:
                    self.pending_actions.pop(user_key, None)
                    pend = None
            if pend and isinstance(pend, dict):
                sel = (incoming_msg or "").strip().lower()
                # Accept simple numeric 1..9 or responses like "opcion 2"
                mnum = re.match(r"^(?:opci[oó]n\s+)?(\d{1,2})\b", sel)
                if mnum:
                    idx = int(mnum.group(1)) - 1
                    events: List[Dict[str, Any]] = pend.get("events", [])
                    if 0 <= idx < len(events):
                        try:
                            ev = events[idx]
                            ev_id = ev.get('id')
                            self.appointments.service.events().delete(calendarId=self.appointments.calendar_id, eventId=ev_id).execute()
                            # Clear pending
                            self.pending_actions.pop(user_key, None)
                            # Proceed with the queued action if any
                            next_action = pend.get("next_action", {})
                            act = (next_action.get("action") or "").lower()
                            when_iso = next_action.get("when_iso")
                            when_dt = None
                            if when_iso:
                                try:
                                    when_dt = datetime.fromisoformat(when_iso).replace(tzinfo=ZoneInfo(self.config.TIMEZONE))
                                except Exception:
                                    when_dt = None
                            if act == "book" and when_dt:
                                res = self.appointments.book_at(user_key, when_dt)
                                return res.get("message", "Listo.")
                            if act == "book":
                                res = self.appointments.book(user_key)
                                return res.get("message", "Listo.")
                            return f"Cita {ev_id} cancelada. ¿Deseas algo más?"
                        except Exception as e:
                            # Do not drop; allow normal flow
                            self.pending_actions.pop(user_key, None)
                    # If invalid selection, reprint options
                    lines = []
                    for i, e in enumerate(pend.get("events", []), start=1):
                        s = e.get('start', {}).get('dateTime')
                        dt = datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(ZoneInfo(self.config.TIMEZONE))
                        lines.append(f"{i}) {dt.strftime('%A %d/%m %H:%M')} (ID: {e['id']})")
                    return "Selecciona 1, 2 o 3 para cancelar y continuar:\n" + "\n".join(lines)
                elif sel in {"no", "n", "cancel", "cancelar"}:
                    self.pending_actions.pop(user_key, None)
                    return "Entendido, no se canceló ninguna cita."
                # If reply doesn't match, fall through to normal processing but keep pending alive
            # Quick path: allow "cancel <ID>" to remove a specific event (supports cap-prompt flow)
            # Quick path: allow "cancel <ID>" to remove a specific event (supports cap-prompt flow)
            if self.appointments:
                m = re.search(r"\bcancel(?:ar)?\s+([A-Za-z0-9_\-@]+)\b", incoming_msg, flags=re.IGNORECASE)
                if m:
                    target_id = m.group(1)
                    # Ensure the event belongs to this user
                    upcoming = self.appointments._upcoming_user_events(user_key)
                    for e in upcoming:
                        if e.get('id') == target_id:
                            try:
                                self.appointments.service.events().delete(calendarId=self.appointments.calendar_id, eventId=target_id).execute()
                                return f"Cita {target_id} cancelada exitosamente."
                            except Exception:
                                break
                    # If not found, continue with normal flow
            # 1) Single model call: route or answer
            history = self.get_or_create_thread(phone_number)
            route = self._route_or_answer(incoming_msg, history)
            if route and route.get("mode") == "appointment" and self.appointments:
                action = (route.get("action") or "").lower()
                when_iso = route.get("when_iso")
                # Parse when_iso if present
                when_dt = None
                if when_iso:
                    try:
                        base = datetime.fromisoformat(when_iso)
                        when_dt = base.replace(tzinfo=ZoneInfo(self.config.TIMEZONE))
                    except Exception:
                        when_dt = self._parse_requested_time(incoming_msg)
                if action == "cancel_all":
                    res = self.appointments.cancel_all(user_key)
                    return res["message"]
                if action == "cancel":
                    if when_dt:
                        # cancel the matching appointment (closest at that time)
                        upcoming = self.appointments._upcoming_user_events(user_key)
                        target_id = None
                        for e in upcoming:
                            s = e.get('start', {}).get('dateTime')
                            if not s:
                                continue
                            sdt = datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(ZoneInfo(self.config.TIMEZONE))
                            # match same day and hour
                            if sdt.date() == when_dt.date() and sdt.hour == when_dt.hour and sdt.minute == when_dt.minute:
                                target_id = e['id']
                                break
                        if target_id:
                            self.appointments.service.events().delete(calendarId=self.appointments.calendar_id, eventId=target_id).execute()
                            return f"Cita {target_id} cancelada exitosamente."
                        # fallback to next
                    res = self.appointments.cancel_next(user_key)
                    return res["message"]
                if action == "check":
                    # List all upcoming instead of only next
                    events = self.appointments.list_upcoming(user_key)
                    if not events:
                        return "No tienes ninguna cita programada."
                    lines = []
                    for e in events:
                        s = e.get('start', {}).get('dateTime')
                        dt = datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(ZoneInfo(self.config.TIMEZONE))
                        lines.append(f"• {dt.strftime('%A %d/%m %H:%M')} (ID: {e['id']})")
                    return "Tus citas programadas:\n" + "\n".join(lines)
                if action == "list":
                    events = self.appointments.list_upcoming(user_key)
                    if not events:
                        return "No tienes ninguna cita programada."
                    lines = []
                    for e in events:
                        s = e.get('start', {}).get('dateTime')
                        dt = datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(ZoneInfo(self.config.TIMEZONE))
                        lines.append(f"• {dt.strftime('%A %d/%m %H:%M')} (ID: {e['id']})")
                    return "Tus citas programadas:\n" + "\n".join(lines)
                if action == "availability":
                    # Suggest next 5 or constrained by weekday if mentioned
                    t = incoming_msg.lower()
                    mapping = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,
                               "lunes":0,"martes":1,"miercoles":2,"miércoles":2,"jueves":3,"viernes":4}
                    day_filter = None
                    for k, v in mapping.items():
                        if k in t:
                            day_filter = v
                            break
                    start_from = self.appointments._now()
                    if any(phrase in t for phrase in ["next week", "proxima semana", "próxima semana", "semana que viene"]):
                        now = start_from
                        days_ahead = (7 - now.weekday()) % 7
                        if days_ahead == 0:
                            days_ahead = 7
                        start_from = (now + timedelta(days=days_ahead)).replace(hour=8, minute=0, second=0, microsecond=0)
                    slots = self.appointments.next_n_slots(5, start_from=start_from, day_filter=day_filter)
                    if not slots:
                        return "No encuentro horarios disponibles en este momento dentro del horario de atención (8:00 a 13:00)."
                    fmt = [s.strftime('%A %d/%m %H:%M') for s in slots]
                    return "Próximos horarios disponibles: " + "; ".join(fmt)
                if action == "book":
                    if when_dt:
                        result = self.appointments.book_at(user_key, when_dt)
                        if result.get("code") == "MONTHLY_CAP":
                            # Present this month's appointments as 1/2/3 and store pending next action
                            month_events = self.appointments._user_events_in_month(user_key)
                            evs = month_events or self.appointments.list_upcoming(user_key)
                            if evs:
                                self.pending_actions[user_key] = {
                                    "type": "cap_cancel",
                                    "events": evs[:3],
                                    "next_action": {"action": "book", "when_iso": when_dt.isoformat()},
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                }
                                lines = []
                                for i, e in enumerate(evs[:3], start=1):
                                    s = e.get('start', {}).get('dateTime')
                                    dt = datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(ZoneInfo(self.config.TIMEZONE))
                                    lines.append(f"{i}) {dt.strftime('%A %d/%m %H:%M')} (ID: {e['id']})")
                                return "Ya tienes 3 citas este mes. Responde 1, 2 o 3 para cancelar una y continuar con tu nueva reserva:\n" + "\n".join(lines)
                        if not result.get("success") and "no está disponible" in (result.get("message", "").lower()):
                            slots = self.appointments.next_n_slots(5, start_from=when_dt)
                            if slots:
                                opts = "; ".join(s.strftime('%A %d/%m %H:%M') for s in slots)
                                result["message"] += f" Opciones cercanas: {opts}."
                    else:
                        result = self.appointments.book(user_key)
                        if result.get("code") == "MONTHLY_CAP":
                            month_events = self.appointments._user_events_in_month(user_key)
                            evs = month_events or self.appointments.list_upcoming(user_key)
                            if evs:
                                self.pending_actions[user_key] = {
                                    "type": "cap_cancel",
                                    "events": evs[:3],
                                    "next_action": {"action": "book"},
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                }
                                lines = []
                                for i, e in enumerate(evs[:3], start=1):
                                    s = e.get('start', {}).get('dateTime')
                                    dt = datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(ZoneInfo(self.config.TIMEZONE))
                                    lines.append(f"{i}) {dt.strftime('%A %d/%m %H:%M')} (ID: {e['id']})")
                                return "Ya tienes 3 citas este mes. Responde 1, 2 o 3 para cancelar una y continuar:\n" + "\n".join(lines)
                    return result["message"]
            # 2) If mode=answer
            if route and route.get("mode") == "answer":
                text = (route.get("text") or "").strip()
                if text:
                    history.append({"role": "user", "content": incoming_msg})
                    history.append({"role": "assistant", "content": text})
                    if len(history) > 2 * self.max_history:
                        del history[: len(history) - 2 * self.max_history]
                    return text
                # Fall back to snippet if empty
                grounding = self._retrieve_context(incoming_msg)
                snippet = self._fallback_from_context(incoming_msg, grounding)
                return snippet or (
                    "No cuento con información suficiente en el documento para responder con precisión. "
                    "¿Puedes reformular tu pregunta o dar más detalles?"
                )

            # 3) If model unavailable
            if not self.openai_client:
                return (
                    "Lo siento, el servicio de respuestas no está disponible en este momento. "
                    "Puedo ayudarte a programar, verificar o cancelar una cita."
                )

            # 4) Router failed: safe fallback to snippet
            grounding = self._retrieve_context(incoming_msg)
            snippet = self._fallback_from_context(incoming_msg, grounding)
            if snippet:
                return snippet
            return (
                "No pude generar una respuesta en este momento. Intenta reformular la pregunta o "
                "pídeme programar, verificar o cancelar una cita."
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "Lo siento, hubo un error procesando tu mensaje. Por favor, intenta de nuevo más tarde."

def create_app(config: Config) -> Flask:
    """Create Flask application"""
    app = Flask(__name__)
    bot = ConsulateBot(config)

    @app.route('/')
    def index() -> Response:
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "message": "Welcome to the Consulate Chatbot"
        })

    @app.route('/', methods=['POST'])
    def webhook_reply() -> str:
        """Unified webhook for SMS/WhatsApp"""
        try:
            # Validate Twilio signature
            signature = request.headers.get('X-Twilio-Signature', '')
            validator = RequestValidator(config.TWILIO_AUTH_TOKEN)
            url = request.url  # full URL of this request
            if not validator.validate(url, request.form, signature):
                logger.warning("Twilio signature validation failed")
                return ("Forbidden", 403)
            incoming_msg = request.values.get('Body', '').strip()
            phone_number = request.values.get('From', '')
            response = MessagingResponse()
            response.message(bot.process_message(phone_number, incoming_msg))
            return str(response)
        except Exception as e:
            logger.error(f"Error in webhook handler: {str(e)}")
            response = MessagingResponse()
            response.message("Lo siento, ocurrió un error. Por favor, intenta de nuevo más tarde.")
            return str(response)

    return app

def main():
    """Application entry point"""
    try:
        config = Config.from_env()
        app = create_app(config)
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise

if __name__ == '__main__':
    main()