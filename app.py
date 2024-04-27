import streamlit as st
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import load_model
# from bert_model import tfidf_vectorizer
import pickle

def load_model():
    model = tf.keras.models.load_model('tfidf_sentiment_model.h5')  # Adjust file path as needed
    return model

# Load TF-IDF vectorizer
def load_tfidf_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as f:  # Adjust file path as needed
        vectorizer = pickle.load(f)
    return vectorizer

# Preprocess text data
def preprocess_text(text, vectorizer):
    text_tfidf = vectorizer.transform([text])
    return text_tfidf

def main():
    st.title("Sentiment Analysis")

    # Load pre-trained deep learning model
    model = load_model()

    # Get user input
    text_input = st.text_input("Enter a sentence for sentiment analysis:")

    if st.button("Analyze Sentiment"):
            # Load TF-IDF vectorizer
        vectorizer = load_tfidf_vectorizer()
        # Preprocess input text
        processed_text = preprocess_text(text_input, vectorizer)

        # Perform sentiment analysis using the pre-trained model
        prediction = model.predict([processed_text])

        # Output sentiment prediction
        if prediction[0] < 0.5:
            st.write("Sentiment: Negative")
        else:
            st.write("Sentiment: Positive")

if __name__ == '__main__':
    main()
