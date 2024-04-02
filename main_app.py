import streamlit as st 
import pickle
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np 
import sklearn


# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("punkt")


# preprocessing data, removing punctuation, tokenizing, lemmatization
def clean_text(text):

    # Remove punctuation and lowercase
    text = ''.join([c for c in text if c not in string.punctuation]).lower()

    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in nltk.word_tokenize(text) if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return str([lemmatizer.lemmatize(token) for token in tokens])


# Loading the trained model and vectorizer
model = pickle.load(open("IMDB_Classification_model.h5", "rb"))  # Load model 
tfidf= pickle.load(open("tfidf_vectorizer.pkl", "rb"))  # load vectorizer


def predict_sentiment(text):
    
    """Predicts sentiment for a given user input.

    Args:
        text: User-provided review text.

    Returns:
        A string representing the predicted sentiment (positive, negative, or neutral).
    """

    try:
        # Clean and vectorize the input
        vectorized_text = tfidf.transform([clean_text(text)]).toarray()


        # Make predictions using the model
        prediction = model.predict(vectorized_text)[0]

        # Simplify sentiment label assignment (assuming 0 is negative, 1 is positive)
        sentiment = "Positive" if prediction == 1 else "Negative"  # Use ternary operator

        return sentiment

    except Exception as e:
        return f"Error: {str(e)}"
  

st.title("Movie Reviews Sentiment Analysis App")

with st.container():
    review_text = st.text_area("Enter your review:", height=100)
    submit_button = st.button("Analyze Sentiment")

if submit_button:
    with st.spinner("Analyzing..."):  # Add a loading spinner
        sentiment = predict_sentiment(review_text)
    st.write(f"Predicted Sentiment: {sentiment}")

# **Footer with disclaimer at the bottom**
st.write("---")  # Add a horizontal line for separation

with st.container():
    st.write(
        "**Disclaimer:** This sentiment analysis is for informational purposes only and should not be taken as absolute truth.  its training data (IMDB reviews) and may not be accurate for all types of text."
    )