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
    OPENAI_ORG_ID: str
    OPENAI_PROJECT_ID: str
    OPENAI_ASSISTANT_ID: str
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    GOOGLE_CALENDAR_ID: Optional[str] = None
    GOOGLE_SERVICE_ACCOUNT_FILE: Optional[str] = None
    TIMEZONE: str = "America/Chicago"
    APPOINTMENT_DURATION_MINUTES: int = 30

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        required_vars = [
            'OPENAI_API_KEY',
            'OPENAI_ORG_ID',
            'OPENAI_PROJECT_ID',
            'OPENAI_ASSISTANT_ID',
            'TWILIO_ACCOUNT_SID',
            'TWILIO_AUTH_TOKEN'
        ]
        
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
            
        return cls(
            OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY'),
            OPENAI_ORG_ID=os.environ.get('OPENAI_ORG_ID'),
            OPENAI_PROJECT_ID=os.environ.get('OPENAI_PROJECT_ID'),
            OPENAI_ASSISTANT_ID=os.environ.get('OPENAI_ASSISTANT_ID'),
            TWILIO_ACCOUNT_SID=os.environ.get('TWILIO_ACCOUNT_SID'),
            TWILIO_AUTH_TOKEN=os.environ.get('TWILIO_AUTH_TOKEN'),
            GOOGLE_CALENDAR_ID=os.environ.get('GOOGLE_CALENDAR_ID'),
            GOOGLE_SERVICE_ACCOUNT_FILE=(
                os.environ.get('GOOGLE_SERVICE_ACCOUNT_FILE')
                or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            ),
            TIMEZONE=os.environ.get('TIMEZONE', 'America/Chicago'),
            APPOINTMENT_DURATION_MINUTES=int(os.environ.get('APPOINTMENT_DURATION_MINUTES', '30'))
        )


class IntentDetector:
    """Lightweight, cheap intent detection using regex/keywords."""

    BOOK_PATTERNS = [
        r"\bappoint(ment|ar)?\b",
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

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client

        
    def classify(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        def any_match(patterns: List[str]) -> bool:
            return any(re.search(p, t) for p in patterns)
        if any_match(self.BOOK_PATTERNS):
            return {"intent": "appointment_request", "confidence": 0.95}
        if any_match(self.CANCEL_PATTERNS):
            return {"intent": "appointment_cancel", "confidence": 0.95}
        if any_match(self.CHECK_PATTERNS):
            return {"intent": "appointment_check", "confidence": 0.9}
        # Fallback to a tiny model if available
        if self.client:
            try:
                comp = self.client.chat.completions.create(
                    model="gpt-5-nano",
                    # use model default temperature
                    max_completion_tokens=20,
                    messages=[
                        {"role": "system", "content": (
                            "Classify this user message intent for a consulate bot. Output strict JSON with keys 'intent' and 'confidence' (0-1). "
                            "Intents: appointment_request, appointment_cancel, appointment_check, general_inquiry. "
                            "User may write in English or Spanish."
                        )},
                        {"role": "user", "content": text[:400]}
                    ]
                )
                raw = comp.choices[0].message.content
                data = json.loads(raw)
                intent = data.get("intent", "general_inquiry")
                conf = float(data.get("confidence", 0.6))
                if intent not in {"appointment_request", "appointment_cancel", "appointment_check", "general_inquiry"}:
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
        # Business hours 09:00-17:00
        while True:
            # Skip weekends
            if cur.weekday() >= 5:
                # move to next Monday 09:00
                days_ahead = 7 - cur.weekday()
                cur = cur + timedelta(days=days_ahead)
                cur = cur.replace(hour=9, minute=0, second=0, microsecond=0)
                continue
            if cur.hour < 9:
                cur = cur.replace(hour=9, minute=0, second=0, microsecond=0)
            if cur.hour >= 17:
                # move to next day 09:00
                cur = (cur + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
                continue
            # Check conflict
            end = cur + self.duration
            if not self._has_conflict(cur, end):
                return cur
            # Move to next slot
            cur = cur + timedelta(minutes=30)

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
        # In-memory message history per user id
        self.threads = {}
        self.max_history = 10  # messages to keep per user
        self.twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        self.intent = IntentDetector(self.openai_client)
        # Quick health check for OpenAI auth; if it fails, disable LLM usage gracefully
        try:
            _ = self.openai_client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "healthcheck"},
                    {"role": "user", "content": "ping"}
                ],
                max_completion_tokens=1,
            )
            logger.info("OpenAI client ready")
        except Exception as e:
            logger.error(f"OpenAI client check failed: {e}. Verify OPENAI_API_KEY and network access.")
            self.openai_client = None
            # Also disable model fallback in intent
            self.intent = IntentDetector(None)
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
                logger.info("Loaded consulate_information.pdf for grounding")
        except Exception as e:
            logger.warning(f"Failed to load PDF for grounding: {e}")

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

    def _retrieve_context(self, query: str, k: int = 6) -> str:
        """Naive keyword retrieval from PDF text to ground answers."""
        if not self.pdf_corpus:
            return ""
        # Split corpus into paragraphs
        paras = [p.strip() for p in self.pdf_corpus.split('\n') if p.strip()]
        q = query.lower()
        scored = []
        for p in paras:
            text = p.lower()
            # simple keyword overlap score
            score = sum(1 for w in re.findall(r"\w+", q) if w in text)
            if score:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        context = "\n".join(p for _, p in scored[:k])
        return context[:4000]

    def _normalize_phone(self, raw: str) -> str:
        """Extract digits so WhatsApp/SMS prefixes don't fragment identity."""
        digits = ''.join(ch for ch in (raw or '') if ch.isdigit())
        if digits:
            return digits
        # Fall back to a short stable hash if we don't have digits (e.g., some channel IDs)
        h = hashlib.sha256((raw or '').encode('utf-8')).hexdigest()
        return h[:24]

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
                    result = self.appointments.book(user_key)
                    return result["message"]
                if intent["intent"] == "appointment_cancel":
                    result = self.appointments.cancel_next(user_key)
                    return result["message"]
                if intent["intent"] == "appointment_check":
                    result = self.appointments.check_next(user_key)
                    return result["message"]

            # 2) Fall back to chat completions for general inquiries
            if not self.openai_client:
                return (
                    "Lo siento, el servicio de respuestas no está disponible en este momento. "
                    "Puedo ayudarte a programar, verificar o cancelar una cita."
                )
            history = self.get_or_create_thread(phone_number)
            grounding = self._retrieve_context(incoming_msg)

            system_persona = (
                "Eres un asistente virtual del consulado. Responde solo a preguntas de servicios consulares "
                "(visados, pasaportes, asistencia legal, etc.). Si no sabes la respuesta o no está en el contexto, "
                "indícalo amablemente. You can speak in both English and Spanish."
            )
            system_context = (
                f"Use ONLY this context to answer. If insufficient, say you don't know.\n{grounding}"
                if grounding else "If no context is provided, say you don't know."
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_persona},
                {"role": "system", "content": system_context},
            ]
            # Append a small rolling history to preserve continuity
            messages.extend(history[-self.max_history:])
            messages.append({"role": "user", "content": incoming_msg})

            comp = self.openai_client.chat.completions.create(
                model="gpt-5-nano",
                # use model default temperature
                max_completion_tokens=300,
                messages=messages,
            )
            reply = comp.choices[0].message.content.strip() if comp.choices else ""
            if reply:
                # Update history
                history.append({"role": "user", "content": incoming_msg})
                history.append({"role": "assistant", "content": reply})
                # trim
                if len(history) > 2 * self.max_history:
                    del history[: len(history) - 2 * self.max_history]
                return reply

            return "Lo siento, no pude generar una respuesta."
            
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