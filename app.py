import os
import json
import traceback
from flask import Flask, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64

app = Flask(__name__)
# Secure session key (using an environment variable if available)
app.secret_key = os.environ.get("SESSION_SECRET", "email-intelligence-hub-fixed-key-123")

# Enable CORS (allowing requests from specific frontend URL if provided)
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://127.0.0.1:8080").rstrip('/')
CORS(app, supports_credentials=True, origins=[FRONTEND_URL, "http://127.0.0.1:8080", "http://localhost:8080"])

# Allow HTTP for development only
if os.environ.get("FLASK_ENV") == "development" or not os.environ.get("RENDER"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
else:
    # Ensure HTTPS for OAuth lib in production
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "0"

# ========================
# ML Model Setup
# ========================
vectorizer = TfidfVectorizer()
model = MultinomialNB()

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "emails_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = vectorizer.fit_transform(df["email"])
        y = df["label"]
        model.fit(X, y)
        print("Model trained successfully.")
    else:
        print(f"Warning: Dataset not found at {csv_path}. ML features will fail.")
except Exception as e:
    print(f"Error loading dataset or training model: {e}")

# ========================
# Google OAuth2 Config
# ========================
FILE_NAME = "client_secret_857155392670-go17su8hn4pfhts5iiqf733gi61g8hfm.apps.googleusercontent.com.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try current folder first (standard for Render deployment)
CLIENT_SECRET_FILE = os.path.join(BASE_DIR, FILE_NAME)

# Try parent folder (for local testing structure)
if not os.path.exists(CLIENT_SECRET_FILE):
    CLIENT_SECRET_FILE = os.path.join(os.path.dirname(BASE_DIR), FILE_NAME)

print(f"Using client secret from: {CLIENT_SECRET_FILE}")

# Custom error handler for better debugging on Render
@app.errorhandler(500)
def handle_500(e):
    error_msg = traceback.format_exc()
    print(f"Server Error:\n{error_msg}")
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e) if app.debug else "Check server logs for details",
        "trace": error_msg if app.debug else None
    }), 500

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

class ConfigurationError(Exception):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

def get_flow(redirect_uri=None):
    """Create OAuth2 flow."""
    uri = redirect_uri or "http://127.0.0.1:5000/oauth2callback"
    
    if (os.environ.get("RENDER") or os.environ.get("FORCE_HTTPS")) and uri.startswith("http://"):
        uri = uri.replace("http://", "https://", 1)
        print(f"DEBUG: Forcing HTTPS redirect URI: {uri}")

    diagnostics = {
        "target_key": "GOOGLE_CLIENT_SECRET_JSON",
        "all_env_keys": sorted(list(os.environ.keys())),
        "files_in_cwd": os.listdir(os.getcwd()) if os.path.exists(os.getcwd()) else [],
        "python_version": os.sys.version,
        "is_render": os.environ.get("RENDER", "False")
    }

    target_key = str(diagnostics["target_key"])
    env_secret = os.environ.get(target_key)
    
    if not env_secret:
        relevant_keys = [k for k in diagnostics["all_env_keys"] if "GOOGLE" in k.upper() or "SECRET" in k.upper()]
        diagnostics["relevant_env_info"] = f"Found {len(relevant_keys)} keys: {relevant_keys}"
        for key in relevant_keys:
            if "GOOGLE" in key.upper() and "SECRET" in key.upper():
                env_secret = os.environ.get(key)
                diagnostics["found_via_fuzzy"] = key
                break

    if env_secret:
        try:
            env_secret = env_secret.strip()
            # If the secret is wrapped in quotes (common mistake), remove them
            if len(env_secret) >= 2 and env_secret.startswith('"') and env_secret.endswith('"'):
                env_secret = env_secret[1:-1].replace('\\"', '"')
            
            client_config = json.loads(env_secret)
            return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=uri)
        except Exception as e:
            raise ConfigurationError(f"Invalid environment secret format: {str(e)}", diagnostics)
    
    if os.path.exists(CLIENT_SECRET_FILE):
        return Flow.from_client_secrets_file(CLIENT_SECRET_FILE, scopes=SCOPES, redirect_uri=uri)
    
    raise ConfigurationError(
        "Google Client Secret not found. Please set 'GOOGLE_CLIENT_SECRET_JSON' env var.",
        diagnostics
    )

# ========================
# API Routes
# ========================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "environment": "production" if os.environ.get("RENDER") else "development",
        "auth": "credentials" in session,
        "email": session.get("user_email", "Not logged in")
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "email" not in data:
            return jsonify({"error": "No email text provided"}), 400
        
        text = data.get("email", "")
        if not text:
            return jsonify({"error": "Empty email text provided"}), 400
        
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        return jsonify({"classe": prediction[0]})
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/login")
def login():
    """Start Google OAuth2 flow."""
    try:
        host_url = request.host_url.rstrip('/')
        if (os.environ.get("RENDER") or os.environ.get("FORCE_HTTPS")) and host_url.startswith("http://"):
            host_url = host_url.replace("http://", "https://", 1)
        
        redirect_uri = f"{host_url}/oauth2callback"
        flow = get_flow(redirect_uri)
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent"
        )
        session["state"] = state
        session["code_verifier"] = flow.code_verifier
        return redirect(authorization_url)
    except ConfigurationError as ce:
        diag_html = "<ul>"
        for k, v in ce.diagnostics.items():
            diag_html += f"<li><b>{k}:</b> {v}</li>"
        diag_html += "</ul>"
        
        return f"""
        <html>
            <body style="font-family: sans-serif; padding: 20px;">
                <h2 style="color: #d93025;">Login Configuration Error</h2>
                <p><b>Message:</b> {str(ce)}</p>
                <div style="background: #f1f3f4; padding: 15px; border-radius: 8px;">
                    <h3>Diagnostic Info:</h3>
                    {diag_html}
                </div>
                <hr>
                <p><b>How to fix:</b> Go to Render Dashboard -> Environment. Ensure you have a variable named <code>GOOGLE_CLIENT_SECRET_JSON</code> with the full JSON content from your client_secret file.</p>
            </body>
        </html>
        """, 500
    except Exception as e:
        app.logger.error(f"Login Init Error: {e}")
        return f"<h2>Unexpected Error</h2><p>{str(e)}</p>", 500

@app.route("/oauth2callback")
def oauth2callback():
    """Handle Google callback."""
    state = session.get("state")
    if not state or state != request.args.get("state"):
        return "<h2>State Mismatch</h2><p>Session might have expired. Please try again.</p>", 400

    try:
        host_url = request.host_url.rstrip('/')
        if os.environ.get("RENDER") and host_url.startswith("http://"):
            host_url = host_url.replace("http://", "https://", 1)
            
        redirect_uri = f"{host_url}/oauth2callback"
        
        flow = get_flow(redirect_uri)
        flow.code_verifier = session.get("code_verifier")
        
        # When fetching token, use the URL as seen by the user (HTTPS on Render)
        auth_response = request.url
        if os.environ.get("RENDER") and auth_response.startswith("http://"):
            auth_response = auth_response.replace("http://", "https://", 1)
            
        flow.fetch_token(authorization_response=auth_response)

        credentials = flow.credentials
        session["credentials"] = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
        }

        # Get user info
        service = build("gmail", "v1", credentials=credentials)
        profile = service.users().getProfile(userId="me").execute()
        session["user_email"] = profile.get("emailAddress", "")

        return redirect(FRONTEND_URL)
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Auth Callback Error:\n{error_details}")
        return f"<h2>Auth Error</h2><p>{str(e)}</p><pre>{error_details if app.debug else ''}</pre>", 500

@app.route("/fetch_emails", methods=["POST"])
def fetch_emails():
    """Fetch real emails via Gmail API."""
    try:
        if "credentials" not in session:
            return jsonify({"error": "Unauthorized. Please login with Google."}), 401

        creds_data = session["credentials"]
        credentials = Credentials(
            token=creds_data["token"],
            refresh_token=creds_data.get("refresh_token"),
            token_uri=creds_data["token_uri"],
            client_id=creds_data["client_id"],
            client_secret=creds_data["client_secret"],
            scopes=creds_data["scopes"],
        )

        service = build("gmail", "v1", credentials=credentials)
        limit = int(request.json.get("limit", 5))

        results = service.users().messages().list(userId="me", maxResults=limit).execute()
        messages = results.get("messages", [])
        
        fetched = []
        for m in messages:
            msg = service.users().messages().get(userId="me", id=m["id"]).execute()
            headers = msg.get("payload", {}).get("headers", [])
            subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
            sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
            
            fetched.append({
                "id": m["id"],
                "subject": subject,
                "sender": sender,
                "snippet": msg.get("snippet", "")
            })

        return jsonify({"emails": fetched})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use PORT from environment for Render/Cloud deployment
    port = int(os.environ.get("PORT", 5000))
    # Production should not have debug=True usually, but keeping it optional
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
