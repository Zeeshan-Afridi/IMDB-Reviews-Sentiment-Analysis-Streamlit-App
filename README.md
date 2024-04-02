Sentiment Analysis App with Streamlit

This Streamlit app provides sentiment analysis for user-entered text. It predicts whether the text expresses a positive, negative, or neutral sentiment.

Features:

1 - Simple and intuitive interface: Users enter their review text in a text area and click a button to get the predicted sentiment.

2 - Pre-trained model: Leverages a pre-trained model (trained on IMDB reviews dataset) for prediction.

3 - Pre-trained vectorizer: It has a pretained TF-IDF Vectorizer trained on 10000 features

4 - Informative disclaimer: Clearly communicates that the predictions are for informational purposes only and have limitations.


Requirements:

Python 3+

-- Streamlit (pip install streamlit)

-- scikit-learn (pip install scikit-learn) (for TfidfVectorizer)

-- nltk (pip install nltk) (for text processing)

-- Pickle (built-in) (for loading the model and vectorizer)


Instructions:

Download the repository.

Install the required libraries: pip install -r requirements.txt

Ensure you have the pre-trained model (saved_model.pkl) and vectorizer (trained_vectorizer.pkl) in the same directory or adjust file paths in the code.

Run the app: streamlit run app.py


