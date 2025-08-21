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
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request, jsonify, Response
from pathlib import Path
from PyPDF2 import PdfReader

# Google Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
    USE_DETERMINISTIC_EXTRACTORS: bool = False

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
            USE_DETERMINISTIC_EXTRACTORS=(os.environ.get('USE_DETERMINISTIC_EXTRACTORS', '0').lower() in {'1','true','yes'})
        )


class IntentDetector:
    """Lightweight, cheap intent detection using regex/keywords."""

    BOOK_PATTERNS = [
        r"\bappoint(ment|ar)?\b",
        r"\bappoint\w{3,}\b",  # catch misspellings like appointmnet
        r"\bappoi?ntm?ent\b",   # common typos
        r"\bappt\b",
        r"\bbook(ing)?\b",
        r"\bschedul(e|ed|ing|ar|ar una|ar la)\b",
        r"\bmake an? appointment\b",
        r"\bset up\b.*\bappointment\b",
        r"\breserv(ar|a|ación|ar cita)\b",
        r"\bagend(ar|ar una|ar la|a)\b",
        r"\bprogram(ar|ar una|ar la|a)\b",
        r"\bquiero (una )?cita\b",
        r"\bnecesito (una )?cita\b",
        r"\bcita( para| de)?\b",
        r"\brenovar pasaporte(.*cita)?\b",
    ]
    CANCEL_PATTERNS = [
        r"\bcancel(ar|la|ado|ación| my)?\b",
        r"\banul(ar|ación)\b",
        r"\bdelete\b.*\bappointment\b",
        r"\bborrar cita\b",
    ]
    CHECK_PATTERNS = [
        r"\bcheck(ing)?\b",
        r"\bverificar\b",
        r"\bcuando es mi cita\b",
        r"\bestado de la cita\b",
        r"\bmi cita\b",
        r"\bwhat time is my appointment\b",
        r"\bwhen is my appointment\b",
        r"\bstatus\b",
    ]
    AVAIL_PATTERNS = [
    r"\bavailable\b",
    r"\bavailability\b",
    # common misspelling
    r"\bavail(?:able|ability|ible)\b",
    # related words users use for openings
    r"\bopen(?:ing|ings)?\b",
    r"\bslots?\b",
    r"\bwhat (times?|hours?)\b",
    r"\bnext week\b",
    r"\bthis week\b",
        r"\bwhen (are|is) (appointments|appt|citas) available\b",
        r"\bhorarios disponibles\b",
    r"\bhoras? disponibles\b",
        r"\bdisponible(s)?\b",
        r"\besta(n)? disponible(s)?\b",
        r"\bhay (citas|horarios)\b",
    r"\bpr(?:ó|o)xima semana\b",
    r"\bsemana que viene\b",
    r"\besta semana\b",
        r"\bes monday available\b",
        r"\besta el lunes disponible\b",
    ]

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client

        
    def classify(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        def any_match(patterns: List[str]) -> bool:
            return any(re.search(p, t) for p in patterns)
        # Prefer availability over booking if user asks about open times
        if any_match(self.AVAIL_PATTERNS):
            return {"intent": "appointment_availability", "confidence": 0.95}
        if any_match(self.BOOK_PATTERNS):
            return {"intent": "appointment_request", "confidence": 0.9}
        if any_match(self.CANCEL_PATTERNS):
            return {"intent": "appointment_cancel", "confidence": 0.95}
        if any_match(self.CHECK_PATTERNS):
            return {"intent": "appointment_check", "confidence": 0.9}
        # Fallback to a tiny model if available
        if self.client:
            try:
                comp = self.client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[
                        {"role": "system", "content": (
                            "Classify this user message intent for a consulate bot. Output strict JSON with keys 'intent' and 'confidence' (0-1). "
                            "Intents: appointment_request, appointment_cancel, appointment_check, appointment_availability, general_inquiry. "
                            "User may write in English or Spanish."
                        )},
                        {"role": "user", "content": text[:400]}
                    ]
                )
                raw = comp.choices[0].message.content
                data = json.loads(raw)
                intent = data.get("intent", "general_inquiry")
                conf = float(data.get("confidence", 0.6))
                if intent not in {"appointment_request", "appointment_cancel", "appointment_check", "appointment_availability", "general_inquiry"}:
                    intent = "general_inquiry"
                return {"intent": intent, "confidence": conf}
            except Exception:
                pass
        return {"intent": "general_inquiry", "confidence": 0.6}


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
                # Fallback: try detect a JSON service account in CWD
                for name in os.listdir('.'):
                    if name.endswith('.json') and 'service' in name.lower():
                        sa_path = name
                        break
            if not sa_path:
                raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_APPLICATION_CREDENTIALS env var, and no service account JSON detected in current directory.")
            logger.info(f"Using Google credentials file at: {os.path.abspath(sa_path)}")
            creds = service_account.Credentials.from_service_account_file(sa_path, scopes=self.scopes)
            # Log the service account email to aid configuration
            try:
                sa_email = getattr(creds, 'service_account_email', None)
                if sa_email:
                    logger.info(f"Google service account: {sa_email}")
            except Exception:
                pass
            return build('calendar', 'v3', credentials=creds, cache_discovery=False)
        except Exception as e:
            logger.error(f"Google Calendar auth/build error: {e}")
            raise

    def _verify_calendar_access(self) -> None:
        """Fail fast if the calendar ID is invalid or not shared with the service account."""
        try:
            info = self.service.calendars().get(calendarId=self.calendar_id).execute()
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
            res = req.execute()
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
        # Enforce monthly limit
        monthly = self._user_events_in_month(user_key)
        if len(monthly) >= 3:
            return {"success": False, "message": "Has alcanzado el límite de 3 citas este mes."}
        if not self._validate_business_time(when):
            return {"success": False, "message": "El horario de atención es de lunes a viernes de 8:00 a.m. a 1:00 p.m."}
        # Round to nearest 30-min slot
        minute = (when.minute // 30) * 30
        when = when.replace(minute=minute, second=0, microsecond=0)
        end = when + self.duration
        if self._has_conflict(when, end):
            return {"success": False, "message": "Ese horario no está disponible. ¿Deseas el siguiente disponible?"}
        created = self.service.events().insert(calendarId=self.calendar_id, body=self._event_payload(user_key, when)).execute()
        return {"success": True, "message": f"Cita confirmada para {when.strftime('%d/%m/%Y %H:%M')}. ID: {created.get('id')}", "event": created}

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
        # Enforce monthly limit of 3 appointments per phone
        monthly = self._user_events_in_month(user_key)
        if len(monthly) >= 3:
            return {"success": False, "message": "Has alcanzado el límite de 3 citas este mes."}
        # Find next available slot and create event
        when = self._next_business_slot()
        body = self._event_payload(user_key, when)
        created = self.service.events().insert(calendarId=self.calendar_id, body=body).execute()
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
        self.service.events().delete(calendarId=self.calendar_id, eventId=target['id']).execute()
        return {"success": True, "message": f"Cita {target['id']} cancelada exitosamente."}

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
        self.threads = {}  # In-memory message history per user id
        self.max_history = 2  # messages to keep per user
        self.twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        self.intent = IntentDetector(self.openai_client)
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
                # Also disable model fallback in intent
                self.intent = IntentDetector(None)
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
                # Pre-build paragraph chunks for better retrieval
                self._pdf_paragraphs = self._build_pdf_paragraphs(self.pdf_corpus)
                logger.info(f"Built {len(self._pdf_paragraphs)} PDF paragraphs for retrieval")
                # Debug: Show first few paragraphs
                for i, para in enumerate(self._pdf_paragraphs[:3]):
                    logger.info(f"PDF para {i}: {para[:200]}...")
            else:
                logger.warning("PDF file not found or empty")
                self._pdf_paragraphs = []
        except Exception as e:
            logger.warning(f"Failed to load PDF for grounding: {e}")
            self._pdf_paragraphs = []

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

    def _build_pdf_paragraphs(self, corpus: str) -> List[str]:
        """Build paragraph-like chunks from raw PDF text.
        Prefer splitting on blank lines; otherwise, create sliding windows of 2-3 lines.
        """
        lines = [ln.rstrip() for ln in corpus.splitlines()]
        # Group by blank lines
        paras: List[str] = []
        cur: List[str] = []
        for ln in lines:
            if not ln.strip():
                if cur:
                    paras.append(" ".join(cur).strip())
                    cur = []
            else:
                cur.append(ln.strip())
        if cur:
            paras.append(" ".join(cur).strip())
        # If the PDF has no blank-line structure, synthesize windows
        if not paras or len(paras) < 5:
            win = 3
            synthesized = []
            for i in range(0, len(lines), win):
                chunk = " ".join(ln.strip() for ln in lines[i:i+win] if ln.strip())
                if chunk:
                    synthesized.append(chunk)
            if synthesized:
                paras = synthesized
        # De-dup short repeats and keep reasonable length
        uniq: List[str] = []
        seen = set()
        for p in paras:
            key = p[:120]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p[:1200])
        return uniq

    def _retrieve_context(self, query: str, k: int = 8) -> str:
        """Score and retrieve the most relevant PDF snippets for grounding.
        Boost currency/fee lines when the user asks about costs or fees.
        Also include neighboring context around top hits to capture headings.
        """
        logger.info(f"Retrieving context for query: '{query}'")
        
        if not getattr(self, "_pdf_paragraphs", None):
            logger.warning("No PDF paragraphs available for retrieval")
            return ""
        
        logger.info(f"Available PDF paragraphs: {len(self._pdf_paragraphs)}")
        # If the PDF is very small (e.g., <= 10 paragraphs, <= 15k chars), just provide the whole corpus.
        # This avoids missing info and still fits within small-model limits.
        try:
            total_chars = len(self.pdf_corpus or "")
        except Exception:
            total_chars = 0
        if total_chars <= 15000:
            logger.info("PDF is small; using full corpus as context")
            return (self.pdf_corpus or "")[:6000]
        
        q = (query or "").lower()
        words = re.findall(r"\w+", q)
        # Expand with simple bilingual synonyms/stems for better matching
        expanded = set(words)
        if any(w in q for w in ["passport", "pasaporte"]):
            expanded.update(["passport", "pasaporte", "pasap"])
        if any(w in q for w in ["renew", "renewal", "renovar", "renovación", "renovacion"]):
            expanded.update(["renew", "renewal", "renov", "renovar", "renovación", "renovacion", "tramitar", "tramite", "trámite", "issue", "issuance", "expedir", "expedicion", "expedición"])
        logger.info(f"Query words expanded: {sorted(list(expanded))[:20]}...")
        
        want_fee = any(w in q for w in [
            "fee", "fees", "cost", "price", "tarifa", "tarifas", "costo", "costos", "valor", "pago", "arancel"
        ])
        passport_terms = ["passport", "pasaporte"]
        renew_terms = ["renew", "renewal", "renovar", "renovación"]
        currency_re = re.compile(r"(US\$|\$|USD|COP)\s?\d[\d,\.]*")
        
        logger.info(f"Want fee: {want_fee}")

        scored: List[tuple[int, int]] = []  # (score, idx)
        for idx, p in enumerate(self._pdf_paragraphs):
            text = p.lower()
            # Base overlap
            base = 0
            for w in expanded:
                if not w:
                    continue
                if w in text:
                    base += 1
                elif len(w) >= 5 and any(stem in text for stem in [w[:5]]):
                    base += 1
            if base == 0:
                # Still consider if paragraph contains strong signals
                base = 1 if (want_fee and currency_re.search(p)) else 0
            score = base
            # Boosts
            if want_fee and currency_re.search(p):
                score += 8
            if any(t in text for t in passport_terms):
                score += 2
            if any(t in text for t in renew_terms):
                score += 2
            # Slight boost for explicit word 'fee schedule' like phrases
            if "fee schedule" in text or "tarif" in text:
                score += 2
            if score > 0:
                scored.append((score, idx))
                logger.info(f"Para {idx} scored {score}: {p[:100]}...")

        logger.info(f"Found {len(scored)} scoring paragraphs")
        
        if not scored:
            logger.warning("No paragraphs scored > 0 for this query")
            # Debug: show a few random paragraphs to see what we have
            for i in range(min(3, len(self._pdf_paragraphs))):
                logger.info(f"Sample para {i}: {self._pdf_paragraphs[i][:200]}...")
            
            # Fallback: return first few paragraphs if nothing scored
            if self._pdf_paragraphs:
                logger.info("Using fallback: returning first few paragraphs")
                fallback_ctx = "\n".join(self._pdf_paragraphs[:5])
                return fallback_ctx[:3500]
            return ""
            
        scored.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Top scoring: {[(s, idx) for s, idx in scored[:3]]}")
        
        # Collect top-k with neighbors
        chosen_idxs = []
        for _, idx in scored[:k]:
            for j in [idx-1, idx, idx+1]:
                if 0 <= j < len(self._pdf_paragraphs):
                    chosen_idxs.append(j)
        # De-dup and keep order
        seen_idx = set()
        chunks: List[str] = []
        for i in sorted(set(chosen_idxs)):
            if i in seen_idx:
                continue
            seen_idx.add(i)
            chunks.append(self._pdf_paragraphs[i])
        # Cap total context size
        ctx = "\n".join(chunks)
        logger.info(f"Retrieved context length: {len(ctx)} chars")
        logger.info(f"Context preview: {ctx[:300]}...")
        return ctx[:3500]

    def _is_passport_renewal_query(self, query: str) -> bool:
        q = (query or "").lower()
        has_passport = ("passport" in q) or ("pasaport" in q)  # matches pasaporte/pasaport-
        # Catch Spanish conjugations: renuevo, renovar, renovación, etc., and English renew/renewal
        renew_signals = ["renew", "renewal", "renov", "tramitar", "trámite", "tramite", "expedir", "expedición", "expedicion", "reexpedir", "sacar"]
        return has_passport and any(sig in q for sig in renew_signals)

    def _extract_passport_renewal_steps(self, context: str, query: str) -> str:
        """Extract enumerated steps for passport application/renewal from context.
        Looks for lines with 1., 2., 3. or semicolon-separated lists in PASAPORTE sections.
        """
        if not context:
            return ""
        is_spanish = any(w in (query or "").lower() for w in ["cómo", "como", "renovar", "pasaporte"])
        # Prefer blocks that mention PASAPORTE or PASSPORT
        blocks = [blk for blk in context.split('\n') if blk]
        prefer = [b for b in blocks if re.search(r"pasaport|passport", b, re.I)] or blocks
        text = "\n".join(prefer[:6])
        # Try to extract enumerated items like 1., 2., 3.
        items = re.findall(r"(?:^|\s)(\d{1,2})\.?\s*([^;\n]+)(?:;|\n|$)", text, flags=re.M)
        if not items:
            # Split by semicolons as fallback
            parts = [p.strip() for p in re.split(r";|\n", text) if p.strip()]
            items = [(str(i+1), part) for i, part in enumerate(parts[:5])]
        if not items:
            return ""
        steps = [desc.strip().rstrip('.') for _, desc in items[:5]]
        if is_spanish:
            return "Requisitos para renovar el pasaporte (según el documento): " + "; ".join(steps) + "."
        return "Passport renewal requirements (from the document): " + "; ".join(steps) + "."

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

    def _parse_requested_time(self, text: str) -> Optional[datetime]:
        """Parse simple requests like 'monday at 1', '8/25 at 1:00 p', 'lunes 1pm'. Returns tz-aware datetime."""
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
        hm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", t)
        if hm:
            hour = int(hm.group(1))
            minute = int(hm.group(2) or 0)
            ampm = (hm.group(3) or "").replace(".", "")
        # Weekday names
        weekdays_en = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        weekdays_es = ["lunes","martes","miercoles","miércoles","jueves","viernes","sabado","sábado","domingo"]
        wd_idx = None
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

    def _is_fee_query(self, query: str) -> bool:
        q = (query or "").lower()
        fee_words = ["fee", "fees", "cost", "price", "tarifa", "tarifas", "costo", "valor", "pago", "arancel"]
        passport_words = ["passport", "pasaporte"]
        renew_words = ["renew", "renewal", "renovar", "renovación"]
        return (any(w in q for w in fee_words) or "$" in q) and any(w in q for w in passport_words + renew_words)

    def _extract_fee_answer(self, query: str, context: str) -> str:
        """Heuristic extraction of fee lines for passport renewal from grounded context.
        Returns a short bilingual answer quoting the exact amount(s).
        """
        if not context:
            return ""
        q = (query or "").lower()
        is_spanish = any(w in q for w in ["cuánto", "renovar", "pasaporte", "tarifa", "costo"])
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
        currency_re = re.compile(r"(US\$|\$|USD|COP)\s?\d[\d,\.]*")
        # score lines that mention passport and renew and contain currency
        scored: List[tuple[int, str]] = []
        for i, ln in enumerate(lines):
            low = ln.lower()
            has_curr = bool(currency_re.search(ln))
            if not has_curr:
                continue
            score = 0
            if "passport" in low or "pasaporte" in low:
                score += 3
            if any(w in low for w in ["renew", "renewal", "renovar", "renovación"]):
                score += 3
            if any(w in low for w in ["fee", "tarifa", "costo", "precio", "valor", "pago", "arancel"]):
                score += 2
            # consider neighbor lines to capture headings or values on separate lines
            neighbor = " ".join(lines[max(0, i-1): min(len(lines), i+2)])
            if currency_re.search(neighbor) and ("passport" in neighbor.lower() or "pasaporte" in neighbor.lower()):
                score += 2
            if score:
                scored.append((score, neighbor.strip()))
        if not scored:
            return ""
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        # Extract all currency mentions in the best snippet
        amounts = currency_re.findall(best)
        # Use the raw neighbor text; keep it concise
        snippet = best
        if is_spanish:
            return f"Según el documento: {snippet}"
        return f"According to the document: {snippet}"

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
            # 1) Try cheap intent detection first
            intent = self.intent.classify(incoming_msg)
            if self.appointments and intent["confidence"] >= 0.8:
                user_key = self._normalize_phone(phone_number)
                if intent["intent"] == "appointment_request":
                    # Try to parse a requested date/time
                    requested = self._parse_requested_time(incoming_msg)
                    if requested:
                        requested_local = requested.astimezone(ZoneInfo(self.config.TIMEZONE))
                        result = self.appointments.book_at(user_key, requested_local)
                        if not result.get("success") and "no está disponible" in (result.get("message", "").lower()):
                            # Offer five nearby alternatives starting from the requested time
                            slots = self.appointments.next_n_slots(5, start_from=requested_local)
                            if slots:
                                opts = "; ".join(s.strftime('%A %d/%m %H:%M') for s in slots)
                                result["message"] += f" Opciones cercanas: {opts}."
                    else:
                        result = self.appointments.book(user_key)
                    return result["message"]
                if intent["intent"] == "appointment_cancel":
                    result = self.appointments.cancel_next(user_key)
                    return result["message"]
                if intent["intent"] == "appointment_check":
                    result = self.appointments.check_next(user_key)
                    return result["message"]
                if intent["intent"] == "appointment_availability":
                    # If the user mentions a weekday, filter; otherwise show next 5 slots
                    t = incoming_msg.lower()
                    mapping = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,
                               "lunes":0,"martes":1,"miercoles":2,"miércoles":2,"jueves":3,"viernes":4}
                    day_filter = None
                    for k, v in mapping.items():
                        if k in t:
                            day_filter = v
                            break
                    # Respect "next week" queries by starting from next Monday 08:00
                    start_from = self.appointments._now()
                    if any(phrase in t for phrase in [
                        "next week", "proxima semana", "próxima semana", "semana que viene"
                    ]):
                        now = start_from
                        # Monday=0 .. Sunday=6 -> compute next Monday
                        days_ahead = (7 - now.weekday()) % 7
                        if days_ahead == 0:
                            days_ahead = 7
                        start_from = (now + timedelta(days=days_ahead)).replace(hour=8, minute=0, second=0, microsecond=0)
                    slots = self.appointments.next_n_slots(5, start_from=start_from, day_filter=day_filter)
                    is_english = any(w in t for w in [
                        "available","availability","opening","openings","slot","what","when","next week","this week",
                        "monday","tuesday","wednesday","thursday","friday"
                    ])
                    if not slots:
                        return (
                            "I can't find available times within service hours (8:00–13:00)." if is_english
                            else "No encuentro horarios disponibles en este momento dentro del horario de atención (8:00 a 13:00)."
                        )
                    fmt = [s.strftime('%A %d/%m %H:%M') for s in slots]
                    return (
                        "Next available times: " + "; ".join(fmt) if is_english
                        else "Próximos horarios disponibles: " + "; ".join(fmt)
                    )

            # 2) Fall back to chat completions for general inquiries
            if not self.openai_client:
                return (
                    "Lo siento, el servicio de respuestas no está disponible en este momento. "
                    "Puedo ayudarte a programar, verificar o cancelar una cita."
                )
            history = self.get_or_create_thread(phone_number)
            grounding = self._retrieve_context(incoming_msg)
            # Short-circuit: if user asks fee for passport renewal and we can extract it from context, answer deterministically
            if self.config.USE_DETERMINISTIC_EXTRACTORS:
                try:
                    if self._is_fee_query(incoming_msg):
                        fee_ans = self._extract_fee_answer(incoming_msg, grounding)
                        if fee_ans:
                            logger.info("Using deterministic fee extractor answer")
                            # Update minimal history for continuity
                            history.append({"role": "user", "content": incoming_msg})
                            history.append({"role": "assistant", "content": fee_ans})
                            if len(history) > 2 * self.max_history:
                                del history[: len(history) - 2 * self.max_history]
                            return fee_ans
                except Exception:
                    pass
            # Short-circuit: passport renewal steps
            if self.config.USE_DETERMINISTIC_EXTRACTORS:
                try:
                    if self._is_passport_renewal_query(incoming_msg):
                        steps_ans = self._extract_passport_renewal_steps(grounding, incoming_msg)
                        if steps_ans:
                            logger.info("Using deterministic passport-steps extractor answer")
                            history.append({"role": "user", "content": incoming_msg})
                            history.append({"role": "assistant", "content": steps_ans})
                            if len(history) > 2 * self.max_history:
                                del history[: len(history) - 2 * self.max_history]
                            return steps_ans
                except Exception:
                    pass

            system_persona = (
                "Eres un asistente virtual del consulado. Responde solo a preguntas de servicios consulares "
                "(visados, pasaportes, asistencia legal, etc.). Si no sabes la respuesta o no está en el contexto, "
                "indícalo amablemente. Responde directo en 1–2 oraciones, sin pasos de razonamiento ni explicaciones largas. "
                "Responde en el mismo idioma del usuario (español o inglés)."
            )
            system_context = (
                f"Use ONLY this context to answer. If details are present, summarize them clearly (steps, requirements, documents, fees) and cite exact amounts as written. If insufficient, say you don't know.\n{grounding}"
                if grounding else "If no context is provided, say you don't know."
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_persona},
                {"role": "system", "content": system_context},
            ]
            # Append a small rolling history to preserve continuity
            messages.extend(history[-self.max_history:])
            messages.append({"role": "user", "content": incoming_msg})

            try:
                comp = self.openai_client.chat.completions.create(
                    model="gpt-5-nano",
                    #max_completion_tokens=350,
                    messages=messages,
                )
            except Exception as e:
                msg = str(e)
                # Retry with trimmed context if token/output limit triggered
                if "max_tokens" in msg or "output limit" in msg:
                    logger.warning("Retrying with trimmed context due to token/output limit")
                    trimmed_ctx = (grounding or "")[:1200]
                    trimmed_messages: List[Dict[str, str]] = [
                        {"role": "system", "content": system_persona},
                        {"role": "system", "content": (
                            f"Use ONLY this context to answer. If insufficient, say you don't know.\n{trimmed_ctx}"
                            if trimmed_ctx else "If no context is provided, say you don't know."
                        )},
                        # keep only the last 2 exchanges max
                        *history[-2:],
                        {"role": "user", "content": incoming_msg},
                    ]
                    comp = self.openai_client.chat.completions.create(
                        model="gpt-5-nano",
                       #max_completion_tokens=300,
                        messages=trimmed_messages,
                    )
                else:
                    raise
            reply = self._extract_text_reply(comp)
            if reply:
                # Update history
                history.append({"role": "user", "content": incoming_msg})
                history.append({"role": "assistant", "content": reply})
                # trim
                if len(history) > 2 * self.max_history:
                    del history[: len(history) - 2 * self.max_history]
                return reply

            # Safe fallback when the model returned no text
            if grounding:
                # Last-resort: surface a snippet directly from the document
                snippet = self._fallback_from_context(incoming_msg, grounding)
                if snippet:
                    history.append({"role": "user", "content": incoming_msg})
                    history.append({"role": "assistant", "content": snippet})
                    if len(history) > 2 * self.max_history:
                        del history[: len(history) - 2 * self.max_history]
                    logger.info("Using context snippet fallback")
                    return snippet
                # Try deterministic extraction for passport steps again as a last resort
                if self.config.USE_DETERMINISTIC_EXTRACTORS:
                    try:
                        if self._is_passport_renewal_query(incoming_msg):
                            steps_ans = self._extract_passport_renewal_steps(grounding, incoming_msg)
                            if steps_ans:
                                history.append({"role": "user", "content": incoming_msg})
                                history.append({"role": "assistant", "content": steps_ans})
                                if len(history) > 2 * self.max_history:
                                    del history[: len(history) - 2 * self.max_history]
                                return steps_ans
                    except Exception:
                        pass
                return (
                    "No cuento con información suficiente en el documento para responder con precisión. "
                    "¿Puedes reformular tu pregunta o dar más detalles?"
                )
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

    @app.route('/whatsapp', methods=['POST'])
    def whatsapp_reply() -> str:
        """Handle incoming WhatsApp messages"""
        try:
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