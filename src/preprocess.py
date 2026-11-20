"""
preprocess.py
--------------
This module contains all the text-cleaning functions used to convert raw,
free-form symptom descriptions (written by a patient in plain English)
into a clean format that a Machine Learning model can understand.

The cleaning pipeline followed here is a classic NLP preprocessing pipeline:
    1. Convert text to lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove special characters
    5. Tokenize the sentence into individual words
    6. Remove stopwords (common words like "is", "the", "and" that add no
       real meaning for classification)
    7. Lemmatize each word (convert words to their base/dictionary form,
       e.g. "running" -> "run", "feet" -> "foot")

These same functions are reused everywhere in the project (training,
prediction and the Streamlit chatbot) so that the text is always cleaned
in exactly the same way.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    """
    Downloads the NLTK resources required for tokenization, stopword
    removal and lemmatization. This only needs to run once - if the
    resources already exist on disk, NLTK will simply skip the download.
    """
    required_resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]
    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


# Make sure NLTK data is available as soon as this module is imported
download_nltk_resources()

# Initialize reusable objects once (instead of creating them again and again)
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words("english"))


def convert_to_lowercase(text):
    """Converts all characters in the text to lowercase."""
    return text.lower()


def remove_punctuation(text):
    """Removes all punctuation marks such as . , ! ? ; : ' \" etc."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    """Removes any digits (0-9) present in the text."""
    return re.sub(r"\d+", "", text)


def remove_special_characters(text):
    """Removes any character that is not a letter or a whitespace."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def remove_extra_whitespace(text):
    """Collapses multiple spaces into a single space and trims the text."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text):
    """Splits a sentence into a list of individual word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Removes common English stopwords (e.g. 'is', 'the', 'and') from a list of tokens."""
    return [word for word in tokens if word not in english_stopwords]


def lemmatize_tokens(tokens):
    """Converts each word in the list to its base dictionary form."""
    return [lemmatizer.lemmatize(word) for word in tokens]


def clean_text(raw_text):
    """
    Runs the complete text-cleaning pipeline on a single piece of text
    and returns a clean string that is ready to be converted into TF-IDF
    features.

    Parameters
    ----------
    raw_text : str
        The original, unprocessed text (e.g. a symptom description
        typed by the patient).

    Returns
    -------
    str
        The fully cleaned and lemmatized text, with words separated by
        single spaces.
    """
    if not isinstance(raw_text, str):
        return ""

    text = convert_to_lowercase(raw_text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)

    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    cleaned_text = " ".join(tokens)
    return cleaned_text


def clean_text_series(text_series):
    """
    Applies clean_text() to an entire Pandas Series (i.e. a full column
    of a DataFrame) and returns a new cleaned Series.

    Parameters
    ----------
    text_series : pandas.Series
        A column containing raw text values.

    Returns
    -------
    pandas.Series
        A column containing the cleaned text values.
    """
    return text_series.apply(clean_text)


# Quick manual test when this file is run directly
if __name__ == "__main__":
    sample_text = "I have been experiencing high Fever, severe joint pain and chills since 3 days!!"
    print("Original text :", sample_text)
    print("Cleaned text  :", clean_text(sample_text))
