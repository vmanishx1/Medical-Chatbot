"""
predict.py
----------
This module loads the trained model and TF-IDF vectorizer from disk and
provides a simple function to predict the most likely disease (along
with a confidence score) for any new symptom description typed by a
user.

This module is used by both the chatbot logic (src/chatbot.py) and can
also be tested directly from the command line.
"""

import os
import sys

import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
TFIDF_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf.pkl")


def load_artifacts(model_path=MODEL_PATH, tfidf_path=TFIDF_PATH):
    """
    Loads the trained model and TF-IDF vectorizer from disk.

    Returns
    -------
    tuple
        (model, tfidf_vectorizer)
    """
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        raise FileNotFoundError(
            "Trained model or vectorizer not found. "
            "Please run 'python src/train.py' first to train and save the model."
        )

    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
    return model, tfidf_vectorizer


def get_confidence_score(model, features):
    """
    Calculates a confidence score (0-100%) for the model's prediction.

    Some models (Logistic Regression, Naive Bayes, Random Forest) support
    predict_proba(), which gives a genuine probability. Linear SVM does
    not support predict_proba() by default, so for that case we fall
    back to using the decision_function() margin, normalized into a
    rough confidence-like percentage.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100
    elif hasattr(model, "decision_function"):
        # decision_function returns a "distance from the boundary" for each class
        scores = model.decision_function(features)[0]
        # Convert raw scores into a probability-like distribution using softmax
        exp_scores = np.exp(scores - np.max(scores))
        softmax_scores = exp_scores / np.sum(exp_scores)
        confidence = np.max(softmax_scores) * 100
    else:
        confidence = None

    return confidence


def predict_disease(symptom_text, model=None, tfidf_vectorizer=None):
    """
    Predicts the most likely disease for a given free-text symptom
    description.

    Parameters
    ----------
    symptom_text : str
        The raw text typed by the patient (e.g. "I have a runny nose and sore throat").
    model : sklearn estimator, optional
        A pre-loaded trained model. If not provided, it will be loaded from disk.
    tfidf_vectorizer : sklearn TfidfVectorizer, optional
        A pre-loaded TF-IDF vectorizer. If not provided, it will be loaded from disk.

    Returns
    -------
    dict
        {
            "predicted_disease": str,
            "confidence": float or None,
            "cleaned_text": str
        }
    """
    if model is None or tfidf_vectorizer is None:
        model, tfidf_vectorizer = load_artifacts()

    cleaned_text = clean_text(symptom_text)

    if cleaned_text.strip() == "":
        return {
            "predicted_disease": None,
            "confidence": None,
            "cleaned_text": cleaned_text,
        }

    features = tfidf_vectorizer.transform([cleaned_text])
    predicted_disease = model.predict(features)[0]
    confidence = get_confidence_score(model, features)

    return {
        "predicted_disease": predicted_disease,
        "confidence": confidence,
        "cleaned_text": cleaned_text,
    }


if __name__ == "__main__":
    # Quick manual test
    sample_inputs = [
        "I have a runny nose, sneezing and a mild sore throat",
        "There are itchy red blisters all over my body and I have a fever",
        "I feel a burning sensation while urinating and need to pee frequently",
    ]

    trained_model, vectorizer = load_artifacts()

    for text in sample_inputs:
        result = predict_disease(text, trained_model, vectorizer)
        print(f"Input: {text}")
        print(f"  -> Predicted Disease : {result['predicted_disease']}")
        if result["confidence"] is not None:
            print(f"  -> Confidence Score  : {result['confidence']:.2f}%")
        print()
