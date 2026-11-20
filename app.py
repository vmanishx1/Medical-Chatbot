"""
app.py
------
Main Streamlit application for the End-to-End Medical Chatbot.

This file is responsible ONLY for the user interface (UI) and chat flow.
All the actual Machine Learning and business logic lives in the src/
folder (preprocess.py, predict.py, chatbot.py) to keep the project
modular and easy to maintain.

Run this app from the project root with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

# Make sure Python can find the modules inside the src/ folder
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from chatbot import get_chatbot_response, load_disease_info
from predict import load_artifacts

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #555555;
        margin-bottom: 25px;
    }
    .result-card {
        background-color: #F4FAFE;
        border: 1px solid #D6EAF8;
        border-left: 6px solid #1B4F72;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .result-card h4 {
        margin-top: 0px;
        color: #1B4F72;
    }
    .warning-card {
        background-color: #FEF9E7;
        border: 1px solid #F9E79F;
        border-left: 6px solid #B7950B;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #7D6608;
    }
    .disclaimer-card {
        background-color: #FDEDEC;
        border: 1px solid #F5B7B1;
        border-left: 6px solid #C0392B;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
        color: #78281F;
        font-size: 0.92rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cache the trained model, vectorizer and disease info so they are only
# loaded once instead of on every single user interaction.
# ----------------------------------------------------------------------
@st.cache_resource
def get_model_and_data():
    """Loads and caches the trained model, TF-IDF vectorizer and disease info table."""
    model, tfidf_vectorizer = load_artifacts()
    disease_info_df = load_disease_info()
    return model, tfidf_vectorizer, disease_info_df


try:
    trained_model, tfidf_vectorizer, disease_info_df = get_model_and_data()
    model_loaded_successfully = True
except FileNotFoundError as error:
    model_loaded_successfully = False
    load_error_message = str(error)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/medical-doctor.png",
        width=80,
    )
    st.markdown("## About this Project")
    st.write(
        "This is an **End-to-End Medical Chatbot** built using classic "
        "Machine Learning (NOT deep learning / LLMs). It predicts a "
        "possible disease based on the symptoms you describe."
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.write(
        "1. Type your symptoms in plain English in the chat box.\n"
        "2. Press **Predict** (or hit Enter).\n"
        "3. View the predicted disease, confidence score, causes, "
        "precautions and recommended doctor.\n"
    )

    st.markdown("---")
    st.markdown("### Tech Stack")
    st.write(
        "- Python, Pandas, NumPy\n"
        "- NLTK (text preprocessing)\n"
        "- Scikit-learn (TF-IDF + ML models)\n"
        "- Streamlit (this UI)\n"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ This chatbot is built for academic / educational purposes only "
        "and is **not** a substitute for professional medical advice."
    )


# ----------------------------------------------------------------------
# Main page header
# ----------------------------------------------------------------------
st.markdown('<p class="main-title">🩺 Medical Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Describe your symptoms below and get an instant, '
    'ML-powered prediction with helpful health information.</p>',
    unsafe_allow_html=True,
)

if not model_loaded_successfully:
    st.error(
        "Trained model files were not found.\n\n"
        f"{load_error_message}\n\n"
        "Please run `python src/train.py` first to train and save the model "
        "before launching this app."
    )
    st.stop()


# ----------------------------------------------------------------------
# Chat history (kept in Streamlit's session state so it persists
# between reruns, but resets when the app restarts)
# ----------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def confidence_badge_html(confidence):
    """Returns a small colored HTML badge based on the confidence score."""
    if confidence is None:
        return ""
    if confidence >= 70:
        color = "#27AE60"   # green - high confidence
    elif confidence >= 40:
        color = "#E67E22"   # orange - medium confidence
    else:
        color = "#C0392B"   # red - low confidence
    return f'<span class="confidence-badge" style="background-color:{color};">{confidence:.1f}% confidence</span>'


def render_bot_response(response):
    """Renders a single chatbot response as a Streamlit chat message."""
    response_type = response.get("type")

    if response_type in ("greeting", "thanks", "empty"):
        st.markdown(response["message"])
        return

    if response_type == "prediction":
        disease = response["predicted_disease"]
        confidence = response["confidence"]

        st.markdown(
            f"""
            <div class="result-card">
                <h4>🦠 Predicted Condition: {disease}</h4>
                {confidence_badge_html(confidence)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if response.get("low_confidence"):
            st.markdown(
                '<div class="warning-card">⚠️ The model is not very confident about '
                "this prediction. Please consider describing your symptoms in more "
                "detail, or consult a doctor directly.</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="result-card">
                <h4>📋 Description</h4>
                <p>{response['description']}</p>
                <h4>🔎 Possible Causes</h4>
                <p>{response['causes']}</p>
                <h4>🛡️ Precautions</h4>
                <p>{response['precautions']}</p>
                <h4>👨‍⚕️ Recommended Doctor</h4>
                <p>{response['doctor']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="disclaimer-card">⚠️ <b>Disclaimer:</b> This prediction is '
            "generated by a Machine Learning model trained on a limited dataset for "
            "academic purposes. It is <b>not a medical diagnosis</b>. Please consult "
            "a certified doctor for accurate diagnosis and treatment.</div>",
            unsafe_allow_html=True,
        )


# Render previous chat history
for chat_entry in st.session_state["chat_history"]:
    with st.chat_message(chat_entry["role"]):
        if chat_entry["role"] == "user":
            st.markdown(chat_entry["content"])
        else:
            render_bot_response(chat_entry["content"])


# ----------------------------------------------------------------------
# Input box + Predict button
# ----------------------------------------------------------------------
st.markdown("#### 💬 Describe your symptoms")

with st.form(key="symptom_form", clear_on_submit=True):
    input_col, button_col = st.columns([5, 1])

    with input_col:
        user_input = st.text_input(
            label="Symptom input box",
            placeholder="e.g. 'I have a runny nose, sore throat and mild fever'",
            label_visibility="collapsed",
        )

    with button_col:
        predict_clicked = st.form_submit_button("🔍 Predict", use_container_width=True)

if predict_clicked and user_input.strip() != "":
    # Add user's message to chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    # Generate chatbot response
    bot_response = get_chatbot_response(
        user_input, trained_model, tfidf_vectorizer, disease_info_df
    )
    st.session_state["chat_history"].append({"role": "assistant", "content": bot_response})

    st.rerun()
elif predict_clicked and user_input.strip() == "":
    st.warning("Please type your symptoms before clicking Predict.")
