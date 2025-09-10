# Consulate Chatbot (Colombia)
### Overview
Flask-based WhatsApp/SMS chatbot that:
* Uses OpenAI (single LLM call) to route between appointment intents and general Q&A.
* Manages Google Calendar appointments (book, list, cancel, availability) with per-user monthly cap (3) and business hours (Mon–Fri 08:00–13:00 local).
* Twilio signature validation (proxy aware) with optional bypass via `SKIP_TWILIO_VALIDATION`.
* PDF grounding (`consulate_information.pdf`) – entire text truncated (~6k chars) supplied to model.

### Key Environment Variables
Required:
```
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
GOOGLE_CALENDAR_ID=your_calendar_id@group.calendar.google.com
GOOGLE_SERVICE_ACCOUNT_FILE=/run/secrets/sa.json
```

Optional:
```
TIMEZONE=America/Chicago
APPOINTMENT_DURATION_MINUTES=30
OPENAI_TIMEOUT=12
SKIP_TWILIO_VALIDATION=1  # ONLY for local testing
PORT=5000
```

### docker-compose (recommended by dev)
`docker-compose.yml` expects `sa.json and .env`:
```
docker compose up --build
```
Place `sa.json` in project root (ignored by git & excluded from image). Compose mounts it at `/run/secrets/sa.json`.

### Running Locally (without Docker)
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
export GOOGLE_SERVICE_ACCOUNT_FILE=./sa.json
python consulate_chatbot.py
```

### Docker Build & Run
Build image:
```
docker build -t consulate-chatbot .
```
Run container (mount service account credentials as sa.json):
```
docker run --rm -p 5000:5000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e TWILIO_ACCOUNT_SID=$TWILIO_ACCOUNT_SID \
  -e TWILIO_AUTH_TOKEN=$TWILIO_AUTH_TOKEN \
  -e GOOGLE_CALENDAR_ID=$GOOGLE_CALENDAR_ID \
  -e GOOGLE_SERVICE_ACCOUNT_FILE=/run/secrets/sa.json \
  -v $(pwd)/sa.json:/run/secrets/sa.json:ro \
  -v $(pwd)/consulate_information.pdf:/app/consulate_information.pdf:ro \
  consulate-chatbot
```

### Twilio Webhook Setup
Set incoming message webhook to:
```
POST https://<your-domain>/
```
Ensure reverse proxy forwards:
```
X-Forwarded-Proto
X-Forwarded-Host
```

### Gunicorn
Container starts with: `gunicorn wsgi:app -b 0.0.0.0:$PORT` (2 workers / 4 threads).

### Appointment Logic
* Business hours: 08:00–13:00 (start times; last valid start 12:30).
* Monthly cap: 3 per user; user picks which to cancel (reply 1/2/3).

### Testing (local)
```
curl -X POST http://localhost:5000/ -d 'From=+15551234567' -d 'Body=Necesito una cita'
```
If signature blocking tests: set `SKIP_TWILIO_VALIDATION=1` in `.env`.

### Update PDF
Replace `consulate_information.pdf` and restart container.

### Common Issues
| Issue | Fix |
|-------|-----|
| 403 Twilio | URL or signature mismatch; ensure HTTPS + headers |
| Calendar auth error | Make sure `sa.json` exists + calendar shared |
| Monthly cap prompt repeats | Reply 1/2/3 or “cancelar” |
| Empty answer | Check PDF present & readable |

### Minimal Deployment (VPS / EC2 / Oracle)
Run container behind Nginx (TLS) → forward to port 5000. Point Twilio to the HTTPS root URL.