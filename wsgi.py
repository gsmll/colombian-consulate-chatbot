# WSGI entrypoint for gunicorn
import os
from consulate_chatbot import create_app, Config

# Load config from environment
config = Config.from_env()
app = create_app(config)

if __name__ == "__main__":
    # For local debug only
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
