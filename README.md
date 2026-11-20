# 🩺 End-to-End Medical Chatbot using Machine Learning

A complete, end-to-end **Medical Chatbot** that predicts a possible disease from a patient's natural-language symptom description, built entirely using **classical Machine Learning** (no LLMs, no deep learning, no external AI APIs). This project was built as a B.Tech 4th-year AI & ML capstone project.

---

## 📌 Project Overview

The chatbot takes a free-text symptom description (e.g. *"I have a runny nose, sore throat and mild fever"*) and:

1. Cleans and preprocesses the text using classic NLP techniques (NLTK).
2. Converts it into numerical features using **TF-IDF Vectorization**.
3. Predicts the most likely disease using a trained **Scikit-learn classifier**.
4. Displays the predicted disease, a **confidence score**, its **description**, **possible causes**, **precautions**, and a **recommended doctor/specialist**.

The entire prediction pipeline runs locally — there is **no dependency on OpenAI, Hugging Face Transformers, LangChain, vector databases, or any deep learning framework.**

> ⚠️ **Disclaimer:** This chatbot is built strictly for academic/educational purposes. It is **not** a certified medical diagnostic tool and should never replace professional medical advice.

---

## ✨ Features

- 🧹 Full NLP preprocessing pipeline (lowercasing, punctuation/number/special character removal, tokenization, stopword removal, lemmatization)
- 📊 In-depth Exploratory Data Analysis (EDA) with visualizations
- 🔢 TF-IDF based feature engineering (unigrams + bigrams)
- 🤖 Training and comparison of **4 ML models**: Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest
- 🏆 Automatic selection of the best-performing model based on F1-score
- 💾 Model persistence using `joblib`
- 💬 Interactive **Streamlit chatbot UI** with chat history, confidence badges, and a "Clear Chat" button
- 📚 A curated `disease_info.csv` reference table (description, causes, precautions, recommended doctor)
- 🗂️ Clean, modular project structure suitable for a final-year capstone submission

---

## 🗂️ Folder Structure

```
Medical-Chatbot/
│
├── data/
│   ├── Symptom2Disease.csv      # Symptom text + disease label dataset
│   └── disease_info.csv         # Disease descriptions, causes, precautions, doctor
│
├── notebooks/
│   └── EDA.ipynb                # Full exploratory data analysis notebook
│
├── models/
│   ├── model.pkl                # Saved best-performing trained model
│   ├── tfidf.pkl                # Saved TF-IDF vectorizer
│   └── model_comparison.csv     # Comparison metrics of all 4 trained models
│
├── src/
│   ├── preprocess.py            # Text cleaning / NLP preprocessing functions
│   ├── train.py                 # Trains, evaluates and saves the best ML model
│   ├── predict.py                # Loads the saved model and predicts disease + confidence
│   └── chatbot.py                # Core chatbot logic (combines prediction + disease info)
│
├── screenshots/                  # App screenshots for documentation
├── app.py                        # Streamlit chatbot web application (UI)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation (this file)
```

---

## ⚙️ Installation

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd Medical-Chatbot
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
This step cleans the data, builds TF-IDF features, trains and compares all 4 models, and saves the best one to `models/model.pkl` and `models/tfidf.pkl`.
```bash
python src/train.py
```

### 5. Run the chatbot
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (usually `http://localhost:8501`) in your browser.

### (Optional) Explore the EDA notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 📊 Dataset

This project uses the **[Symptom2Disease dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)** from Kaggle, which contains natural-language symptom descriptions mapped to **24 disease categories** such as Common Cold, Dengue, Typhoid, Migraine, Diabetes, Urinary Tract Infection, and more.

| Column | Description |
|--------|-------------|
| `label` | The disease name (target variable) |
| `text`  | A natural-language description of symptoms, written as a patient would type it |

> **Note:** Since this project was developed in an offline/sandboxed environment without direct access to Kaggle, `data/Symptom2Disease.csv` included here is a **synthetically generated dataset that exactly follows the same schema and disease categories** as the original Kaggle dataset, so that the full pipeline can run end-to-end. **For your actual submission, simply download the real dataset from the Kaggle link above and replace `data/Symptom2Disease.csv` — no code changes are required**, since column names (`label`, `text`) are identical.

A second file, `data/disease_info.csv`, was manually curated for this project to power the chatbot's responses. It contains:

| Column | Description |
|--------|-------------|
| `Disease` | Disease name (matches the `label` column above) |
| `Description` | A short explanation of the condition |
| `Causes` | Common causes/risk factors |
| `Precautions` | Recommended precautions |
| `Recommended Doctor` | The type of specialist to consult |

---

## 🧠 Machine Learning Pipeline

1. **Text Preprocessing** (`src/preprocess.py`): lowercasing → punctuation removal → number removal → special character removal → tokenization → stopword removal → lemmatization.
2. **Feature Engineering**: TF-IDF vectorization with unigrams + bigrams (max 3000 features).
3. **Model Training** (`src/train.py`): trains and evaluates 4 classical ML models on an 80/20 train-test split.
4. **Model Selection**: the model with the highest **weighted F1-score** is automatically selected and saved.
5. **Prediction** (`src/predict.py`): cleans new input text, transforms it with the saved TF-IDF vectorizer, and predicts the disease with a confidence score.
6. **Chatbot Logic** (`src/chatbot.py`): wraps the prediction step with disease information lookup and simple conversational handling (greetings, thanks, empty input).

---

## 🖼️ Screenshots

> Screenshots should be added to the `screenshots/` folder after running the app locally. See `screenshots/README.txt` for exact filenames to use.

| Screen | Preview |
|--------|---------|
| Chat Interface |  `screenshots/1-chat-interface.png` |
| Prediction Result | `screenshots/2-prediction-result.png` |
| EDA Word Cloud | `screenshots/3-eda-wordcloud.png` |
| Model Comparison | `screenshots/4-model-comparison.png` |

---

## 🏆 Results

Model comparison results obtained on the included dataset (80/20 train-test split):

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| **Multinomial Naive Bayes** ✅ | 0.9906 | 0.9925 | 0.9906 | **0.9904** |
| Linear SVM | 0.9906 | 0.9925 | 0.9906 | 0.9904 |
| Logistic Regression | 0.9811 | 0.9874 | 0.9811 | 0.9817 |
| Random Forest | 0.9717 | 0.9770 | 0.9717 | 0.9713 |

✅ = automatically selected as the best model and saved to `models/model.pkl`

> These numbers were generated on the bundled dataset and will vary once you replace it with the real Kaggle Symptom2Disease dataset (which is larger and more diverse), but the same pipeline and code will apply without changes.

---

## 🚀 Future Scope

- Integrate **multi-symptom severity scoring** (e.g. mild/moderate/severe) for more nuanced triage.
- Add **multi-language support** for symptom input (e.g. Hindi, Telugu) using translation preprocessing.
- Add a **feedback loop** where users can confirm/correct predictions to improve the model over time (active learning).
- Deploy the Streamlit app on **Streamlit Community Cloud / Render / Heroku** for public access.
- Extend `disease_info.csv` with more diseases and richer medical content (verified by a medical professional).
- Add **voice input** support for accessibility.
- Introduce a simple **rule-based symptom checklist** alongside free-text input for users who prefer structured input.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|----------|--------------------|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| NLP | NLTK (tokenization, stopwords, lemmatization) |
| Machine Learning | Scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Linear SVM, Random Forest) |
| Model Persistence | Joblib |
| Web App / UI | Streamlit |

**Deliberately NOT used:** OpenAI API, LangChain, LlamaIndex, Pinecone, FAISS, RAG, Hugging Face Transformers, TensorFlow, PyTorch — this project demonstrates that a genuinely useful chatbot can be built using only classical, interpretable Machine Learning techniques.

---

## 📄 License

This project is created for academic purposes as part of a B.Tech AI & ML curriculum. Feel free to fork, modify, and build upon it for your own learning.
