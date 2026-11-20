"""
chatbot.py
----------
This module contains the core "brain" of the Medical Chatbot. It combines:
    1. The disease prediction logic (predict.py)
    2. The disease information lookup (disease_info.csv)

into a single, easy-to-use function that the Streamlit app (app.py) can
call directly. Keeping this logic separate from app.py makes the project
more modular and easier to test/maintain.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import load_artifacts, predict_disease

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISEASE_INFO_PATH = os.path.join(PROJECT_ROOT, "data", "disease_info.csv")

# Minimum confidence (%) below which we warn the user that the prediction
# might not be very reliable.
LOW_CONFIDENCE_THRESHOLD = 40

# A simple list of words that hint the input is not actually a symptom
# description (e.g. greetings, thanks, etc.) - used to make the chatbot
# feel slightly more natural without needing any advanced NLP.
GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
THANKS_KEYWORDS = ["thanks", "thank you", "thank you so much", "ty"]


def load_disease_info(path=DISEASE_INFO_PATH):
    """Loads the disease_info.csv file into a Pandas DataFrame."""
    disease_info_df = pd.read_csv(path)
    return disease_info_df


def get_disease_details(disease_name, disease_info_df):
    """
    Looks up the description, causes, precautions and recommended doctor
    for a given disease name.

    Parameters
    ----------
    disease_name : str
        The disease predicted by the ML model.
    disease_info_df : pandas.DataFrame
        The loaded disease_info.csv data.

    Returns
    -------
    dict
        A dictionary containing description, causes, precautions and
        recommended doctor. If the disease is not found, default
        placeholder text is returned instead.
    """
    matching_rows = disease_info_df[disease_info_df["Disease"] == disease_name]

    if matching_rows.empty:
        return {
            "description": "Detailed information for this condition is not available yet.",
            "causes": "Not available.",
            "precautions": "Please consult a certified doctor for accurate guidance.",
            "doctor": "General Physician",
        }

    row = matching_rows.iloc[0]
    return {
        "description": row["Description"],
        "causes": row["Causes"],
        "precautions": row["Precautions"],
        "doctor": row["Recommended Doctor"],
    }


def is_greeting(user_text):
    """Checks whether the user's message is just a greeting rather than a symptom description."""
    text = user_text.lower().strip()
    return any(text.startswith(word) for word in GREETING_KEYWORDS)


def is_thanks(user_text):
    """Checks whether the user's message is a thank-you / closing remark."""
    text = user_text.lower().strip()
    return any(word in text for word in THANKS_KEYWORDS)


def get_chatbot_response(user_text, model, tfidf_vectorizer, disease_info_df):
    """
    Generates a complete chatbot response for a given user message.

    Parameters
    ----------
    user_text : str
        The raw message typed by the user in the Streamlit chat input box.
    model : sklearn estimator
        The trained classification model (already loaded).
    tfidf_vectorizer : sklearn TfidfVectorizer
        The trained TF-IDF vectorizer (already loaded).
    disease_info_df : pandas.DataFrame
        The loaded disease_info.csv data.

    Returns
    -------
    dict
        A dictionary describing the type of response and the content
        needed to render it on the Streamlit chat UI. Possible "type"
        values are: "greeting", "thanks", "empty", "low_confidence",
        "prediction".
    """
    if not user_text or user_text.strip() == "":
        return {"type": "empty"}

    if is_greeting(user_text):
        return {
            "type": "greeting",
            "message": "Hello! I'm your Medical Assistant Chatbot. "
                       "Please describe your symptoms in detail (for example: "
                       "'I have a runny nose, sore throat and mild fever') "
                       "and I will try to identify a possible condition.",
        }

    if is_thanks(user_text):
        return {
            "type": "thanks",
            "message": "You're welcome! Take care of your health, and don't "
                       "hesitate to consult a doctor if your symptoms persist.",
        }

    # Run the ML prediction pipeline
    result = predict_disease(user_text, model, tfidf_vectorizer)
    predicted_disease = result["predicted_disease"]
    confidence = result["confidence"]

    if predicted_disease is None:
        return {
            "type": "empty",
            "message": "I couldn't understand any specific symptoms in your "
                       "message. Could you please describe what you are feeling "
                       "in a bit more detail?",
        }

    disease_details = get_disease_details(predicted_disease, disease_info_df)

    response = {
        "type": "prediction",
        "predicted_disease": predicted_disease,
        "confidence": confidence,
        "description": disease_details["description"],
        "causes": disease_details["causes"],
        "precautions": disease_details["precautions"],
        "doctor": disease_details["doctor"],
        "low_confidence": (confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD),
    }
    return response


if __name__ == "__main__":
    # Quick manual test of the full chatbot pipeline
    trained_model, vectorizer = load_artifacts()
    info_df = load_disease_info()

    test_messages = [
        "hello",
        "I have severe joint pain, stiffness and swelling in my knees",
        "thanks a lot",
    ]

    for msg in test_messages:
        print(f"User: {msg}")
        bot_response = get_chatbot_response(msg, trained_model, vectorizer, info_df)
        print(f"Bot : {bot_response}")
        print()
