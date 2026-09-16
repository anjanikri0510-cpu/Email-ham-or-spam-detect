"""
Streamlit app that serves predictions from spam_pipeline.pkl
(sklearn Pipeline: CountVectorizer(preprocessor=wordopt) -> RandomForestClassifier)

Run:
    pip install streamlit scikit-learn joblib
    streamlit run streamlit_app.py
"""

import re
import string
import sys

import joblib
import streamlit as st

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


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def predict_message(pipeline, message):
    label = int(pipeline.predict([message])[0])
    proba = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([message])[0]
        classes = list(pipeline.classes_)
        proba = float(probs[classes.index(label)])
    return label, proba


st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Spam Classifier")
st.write("Paste a message below and check whether it's spam.")

pipeline = load_pipeline()

message = st.text_area("Message", height=150, placeholder="Type or paste a message here...")

if st.button("Check", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        label, proba = predict_message(pipeline, message)
        if label == 1:
            st.error(f"🚨 **SPAM**" + (f" — {proba * 100:.1f}% confidence" if proba is not None else ""))
        else:
            st.success(f"✅ **NOT SPAM**" + (f" — {proba * 100:.1f}% confidence" if proba is not None else ""))
