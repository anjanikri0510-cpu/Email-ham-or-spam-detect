"""
Flask app that serves predictions from spam_pipeline.pkl
(sklearn Pipeline: CountVectorizer(preprocessor=wordopt) -> RandomForestClassifier)

Run:
    pip install flask scikit-learn joblib
    python app.py
Then open http://localhost:5000
"""

import re
import string
import sys

import joblib
from flask import Flask, render_template_string, request, jsonify

# ---------------------------------------------------------------------------
# IMPORTANT: the pickle was saved with a custom preprocessing function called
# "wordopt" that lived in __main__ at save time. joblib/pickle looks it up by
# name in __main__ when unpickling, so we must define it here, in __main__,
# with the same name, BEFORE loading the pipeline.
# ---------------------------------------------------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Make it discoverable as __main__.wordopt regardless of how this file is run
sys.modules["__main__"].wordopt = wordopt

MODEL_PATH = "spam_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spam Classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; color: #222; }
    h1 { font-size: 1.4rem; }
    textarea { width: 100%; height: 140px; font-size: 1rem; padding: 10px; box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 20px; font-size: 1rem; cursor: pointer; }
    .result { margin-top: 20px; padding: 16px; border-radius: 8px; font-weight: bold; }
    .spam { background: #fdecea; color: #b3261e; }
    .ham { background: #e6f4ea; color: #1e7b34; }
  </style>
</head>
<body>
  <h1>Spam Classifier</h1>
  <form method="post">
    <textarea name="message" placeholder="Paste a message to classify...">{{ message or "" }}</textarea><br>
    <button type="submit">Check</button>
  </form>
  {% if label is not none %}
    <div class="result {{ 'spam' if label == 1 else 'ham' }}">
      Prediction: {{ "SPAM" if label == 1 else "NOT SPAM" }}
      {% if proba is not none %}
        <br><span style="font-weight: normal;">Confidence: {{ "%.1f"|format(proba * 100) }}%</span>
      {% endif %}
    </div>
  {% endif %}
</body>
</html>
"""


def predict_message(message):
    label = int(pipeline.predict([message])[0])
    proba = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([message])[0]
        classes = list(pipeline.classes_)
        proba = float(probs[classes.index(label)])
    return label, proba


@app.route("/", methods=["GET", "POST"])
def index():
    message, label, proba = None, None, None
    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            label, proba = predict_message(message)
    return render_template_string(PAGE, message=message, label=label, proba=proba)


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint.
    Request:  {"message": "some text"}
    Response: {"message": "...", "label": 0 or 1, "prediction": "spam"/"not spam", "confidence": 0.0-1.0}
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400

    label, proba = predict_message(message)
    return jsonify(
        {
            "message": message,
            "label": label,
            "prediction": "spam" if label == 1 else "not spam",
            "confidence": proba,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
