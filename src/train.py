"""
train.py
--------
This script trains and compares four different Machine Learning models
on the Symptom2Disease dataset and automatically saves the best
performing model (along with the TF-IDF vectorizer used to create the
features) to the models/ folder.

Models compared:
    1. Multinomial Naive Bayes
    2. Logistic Regression
    3. Linear Support Vector Machine (SVM)
    4. Random Forest

Run this script from the project root with:
    python src/train.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Allow this script to import preprocess.py whether it is run from the
# project root (python src/train.py) or from inside the src/ folder.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text_series

# ----------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Symptom2Disease.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path):
    """Loads the Symptom2Disease dataset from a CSV file."""
    df = pd.read_csv(path)

    # The original Kaggle file contains an unwanted index column - drop it if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.dropna(subset=["label", "text"])
    df = df.drop_duplicates(subset=["text"])
    return df


def build_features(df):
    """Cleans the raw text and converts it into TF-IDF features."""
    print("Cleaning text data ...")
    df["cleaned_text"] = clean_text_series(df["text"])

    print("Building TF-IDF features ...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,   # limit vocabulary size to keep the model light
        ngram_range=(1, 2),  # use both single words and pairs of words
        sublinear_tf=True,
    )
    X = tfidf_vectorizer.fit_transform(df["cleaned_text"])
    y = df["label"]
    return X, y, tfidf_vectorizer


def get_candidate_models():
    """Returns a dictionary of the four models we want to train and compare."""
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    return models


def evaluate_model(model_name, model, X_test, y_test):
    """
    Calculates and prints accuracy, precision, recall, F1-score,
    classification report and confusion matrix for a single model.

    Returns
    -------
    dict
        A dictionary holding the evaluation metrics for this model.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model_name": model_name,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Trains all candidate models and returns evaluation results for each one."""
    results = []
    models = get_candidate_models()

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    return results


def select_best_model(results):
    """Picks the model with the highest F1-score (a balanced metric for multi-class problems)."""
    best_result = max(results, key=lambda r: r["f1_score"])
    return best_result


def print_comparison_table(results):
    """Prints a simple side-by-side comparison table of all trained models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    comparison_df = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Accuracy": round(r["accuracy"], 4),
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1-score": round(r["f1_score"], 4),
            }
            for r in results
        ]
    ).sort_values(by="F1-score", ascending=False)
    print(comparison_df.to_string(index=False))
    return comparison_df


def main():
    print("Loading dataset ...")
    df = load_data(DATA_PATH)
    print(f"Dataset shape after cleaning duplicates/missing values: {df.shape}")

    X, y, tfidf_vectorizer = build_features(df)

    print("\nSplitting dataset into train and test sets (80/20 split) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = train_and_compare_models(X_train, X_test, y_train, y_test)
    comparison_df = print_comparison_table(results)

    best_result = select_best_model(results)
    print(f"\nBest model selected: {best_result['model_name']} "
          f"(F1-score = {best_result['f1_score']:.4f})")

    # Save the comparison table for use in the README/EDA notebook
    comparison_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)

    # Save the best model and the TF-IDF vectorizer
    joblib.dump(best_result["model"], MODEL_PATH)
    joblib.dump(tfidf_vectorizer, TFIDF_PATH)

    print(f"\nSaved best model to   : {MODEL_PATH}")
    print(f"Saved TF-IDF vectorizer to : {TFIDF_PATH}")


if __name__ == "__main__":
    main()
