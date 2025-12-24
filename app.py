# =========================================
# Spam Email Detection Web App
# =========================================

import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# =========================================
# Load Saved Models & Vectorizer
# =========================================
nb_model = joblib.load("naive_bayes.pkl")
lr_model = joblib.load("logistic_regression.pkl")
svm_model = joblib.load("svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =========================================
# Text Cleaning Function
# =========================================
def clean_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# =========================================
# Streamlit App Layout
# =========================================
st.set_page_config(page_title="Spam Email Detector", page_icon="✉️")
st.title("📧 Spam Email Detection App")
st.write("Enter an email or message below to check if it's spam or not.")

# User Input
user_input = st.text_area("Your Email/Text Here:")

# Model Selection
model_choice = st.selectbox(
    "Choose Model:",
    ("Logistic Regression", "Naive Bayes", "Support Vector Machine")
)

# Prediction Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction!")
    else:
        # Preprocess
        cleaned_text = clean_text(user_input)
        text_vec = vectorizer.transform([cleaned_text])

        # Choose Model
        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(text_vec)
        elif model_choice == "Naive Bayes":
            prediction = nb_model.predict(text_vec)
        else:
            prediction = svm_model.predict(text_vec)

        # Display Result
        result = "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam"
        st.success(f"Prediction: {result}")
