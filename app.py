import os
import json
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
# Secure session key (using a fixed key for development to persist sessions)
app.secret_key = "email-intelligence-hub-fixed-key-123"

# Enable CORS (allowing requests from ANY frontend, e.g., AWS VM)
CORS(app, supports_credentials=True)

# Allow HTTP for development (required for OAuth lib on non-HTTPS)
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# ========================
# ML Model Setup
# ========================
vectorizer = TfidfVectorizer()
model = MultinomialNB()

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "emails_dataset.csv")
    df = pd.read_csv(csv_path)
    X = vectorizer.fit_transform(df["email"])
    y = df["label"]
    model.fit(X, y)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error loading dataset or training model: {e}")

# ========================
# Google OAuth2 Config
# ========================
# Path resolution: find client_secret file in same folder or parent
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
    import traceback
    error_msg = traceback.format_exc()
    print(f"Server Error:\n{error_msg}")
    return f"<h2>Server Error (500)</h2><pre>{error_msg}</pre>", 500

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def get_flow(redirect_uri=None):
    """Create OAuth2 flow."""
    if not os.path.exists(CLIENT_SECRET_FILE):
        raise FileNotFoundError(f"Google Client Secret file not found at: {CLIENT_SECRET_FILE}. Please ensure it is uploaded to the 'backend/' folder on Render.")
    
    # Priority 1: Use Render redirect URI if specified
    # Priority 2: Use local fallback
    uri = redirect_uri or "http://127.0.0.1:5000/oauth2callback"
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=uri
    )
    return flow

# ========================
# API Routes
# ========================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "auth": "credentials" in session,
        "email": session.get("user_email", "Not logged in")
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.json.get("email", "")
        if not text:
            return jsonify({"error": "No email text provided"}), 400
        
        # ML Prediction (no manual 'unknown' checks, let the model decide)
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        return jsonify({"classe": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login")
def login():
    """Start Google OAuth2 flow."""
    # Detect if we should redirect back to Render or Local
    host = request.host_url.rstrip('/')
    redirect_uri = f"{host}/oauth2callback"
    
    flow = get_flow(redirect_uri)
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    session["state"] = state
    # Store the code_verifier for PKCE in the session
    session["code_verifier"] = flow.code_verifier
    return redirect(authorization_url)

@app.route("/oauth2callback")
def oauth2callback():
    """Handle Google callback."""
    if "state" not in session or session["state"] != request.args.get("state"):
        return "<h2>State Mismatch</h2><p>Session might have expired. Please try again.</p>", 400

    try:
        host = request.host_url.rstrip('/')
        redirect_uri = f"{host}/oauth2callback"
        
        flow = get_flow(redirect_uri)
        # Restore the code_verifier from session
        flow.code_verifier = session.get("code_verifier")
        flow.fetch_token(authorization_response=request.url)

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

        # Instead of a success message, redirect back to the frontend!
        # Local development defaults to 127.0.0.1:8080
        frontend_url = "http://127.0.0.1:8080"
        return redirect(frontend_url)
    except Exception as e:
        return f"<h2>Auth Error</h2><p>{str(e)}</p>", 500

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
    app.run(host="0.0.0.0", port=5000, debug=True)
