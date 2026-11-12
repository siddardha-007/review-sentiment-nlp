import streamlit as st
import joblib
import numpy as np
from src.preprocess import clean_text

st.set_page_config(page_title="Flipkart Review Sentiment Analyzer", page_icon="🛍️")

@st.cache_resource
def load_model():
    return joblib.load("models/best_sentiment_model.pkl")

model = load_model()

st.title("Flipkart Product Review Sentiment Analyzer")
st.write("Paste any product review below to predict customer sentiment using **TF-IDF + LinearSVC**.")

review_input = st.text_area(
    "Product Review:",
    placeholder="e.g., Battery life is superb and display is crisp, though charger feels slightly cheap."
)

if st.button("Predict Sentiment"):
    if review_input.strip():
        cleaned = clean_text(review_input)
        prediction = model.predict([cleaned])[0]
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_

        # Display result banner
        if prediction == "Positive":
            st.success(f"**Predicted Sentiment:** {prediction}")
        elif prediction == "Neutral":
            st.warning(f"**Predicted Sentiment:** {prediction}")
        else:
            st.error(f"**Predicted Sentiment:** {prediction}")

        # Show class probability breakdown
        st.subheader("Confidence Breakdown")
        for cls, prob in zip(classes, probs):
            st.write(f"**{cls}:** {prob * 100:.1f}%")
            st.progress(float(prob))

        # Show which words from the review were found in the TF-IDF vocabulary
        vectorizer = model.named_steps["tfidf"]
        vocab = vectorizer.vocabulary_
        matched_terms = [w for w in cleaned.split() if w in vocab]
        if matched_terms:
            st.caption(f"**Key recognised words analysed:** {', '.join(set(matched_terms))}")
    else:
        st.info("Please type or paste a review first.")