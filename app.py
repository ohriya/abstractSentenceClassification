# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import pandas as pd
import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import joblib
from scipy.sparse import hstack
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK stopwords data
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_abstract_text(df):
    return df

def main():
    st.title("Abstract Level Sentence Classification")

    # Text input for the user to enter the abstract
    abstract_input = st.text_area("Enter the abstract:", "")

    if st.button("Classify"):
        # Create a DataFrame with the abstract lines
        abstract_lines = abstract_input.split(".")
        abstract_lines = [line.strip() for line in abstract_lines if line.strip()]
        df = pd.DataFrame({'abstract_text': abstract_lines})
        df['total_lines'] = len(df)
        df['line_number'] = df.index

        # Preprocess abstract
        df = preprocess_abstract_text(df)

        # Load models
        vectorizer = joblib.load('text_vectorizer.joblib')
        nb_model = joblib.load('naive_bayes.joblib')

        # Feature extraction
        inference_numerical = df[['line_number', 'total_lines']].values
        inference_bow = vectorizer.transform(df['abstract_text'])  # No need to use lemmatized_column
        inference_final = hstack((inference_numerical, inference_bow), format='csr')

        # Make predictions
        predictions = nb_model.predict(inference_final)
        df['predictions'] = predictions

        # Display results in Streamlit
        st.write("Predictions:")
        st.table(df[['abstract_text', 'predictions']])
if __name__ == "__main__":
    main()
