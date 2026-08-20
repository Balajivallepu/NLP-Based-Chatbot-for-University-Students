"""
app.py
------
Flask web app serving the trained intent-classification chatbot
with analytics logging, user feedback loop, and topic discovery.
"""
import os
import re
import json
import time
import uuid
import string
import sys
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
CHAT_LOG_FILE = os.path.join(OUTPUTS_DIR, "chat_history.jsonl")

app = Flask(__name__)

# ---------------- Load artifacts once at startup ----------------
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "intent_classifier.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
with open(os.path.join(MODELS_DIR, "response_map.json"), "r", encoding="utf-8") as f:
    response_map = json.load(f)

CONFIDENCE_THRESHOLD = 0.35

# Import the centralized clean_text utility
sys.path.append(BASE_DIR)
from preprocess import clean_text


def get_reply(message: str):
    cleaned = clean_text(message)
    if not cleaned:
        return (
            "I couldn't detect a clear question. Please try asking about admissions, fees, hostel, exams, syllabus, or placements!",
            None,
            0.0
        )

    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    intent = label_encoder.inverse_transform([top_idx])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        fallback_msg = (
            "I'm not fully sure what you're asking. Could you please rephrase, or choose from one of the topics: "
            "Admissions, Fees, Exams, Hostel, Library, or Placements?"
        )
        return fallback_msg, None, confidence

    reply = response_map[intent]["response"]
    return reply, intent, confidence


def log_interaction(entry: dict):
    """Append student query/feedback to jsonl log for analytics."""
    try:
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Logging error: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = data.get("message", "").strip()
    msg_id = str(uuid.uuid4())

    reply, intent, confidence = get_reply(message)

    category = response_map.get(intent, {}).get("category", "General") if intent else "Uncertain"

    log_interaction({
        "id": msg_id,
        "timestamp": datetime.now().isoformat(),
        "query": message,
        "intent": intent,
        "confidence": round(confidence, 4),
        "category": category
    })

    return jsonify({
        "id": msg_id,
        "reply": reply,
        "intent": intent,
        "confidence": round(confidence, 3),
        "category": category
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True) or {}
    msg_id = data.get("id")
    is_positive = data.get("helpful", True)
    
    log_interaction({
        "type": "feedback",
        "id": msg_id,
        "timestamp": datetime.now().isoformat(),
        "helpful": is_positive
    })
    
    return jsonify({"status": "success", "message": "Feedback recorded. Thank you!"})


@app.route("/api/intents", methods=["GET"])
def get_intents():
    """Return all available intents and sample questions for frontend discovery."""
    categories = {}
    for intent, data in response_map.items():
        cat = data.get("category", "General")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "intent": intent,
            "sample_response": data.get("response")
        })
    return jsonify({"total_intents": len(response_map), "categories": categories})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "intents_loaded": len(response_map)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
