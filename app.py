import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('cv.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# Function to predict stress level
def predict_stress(text):
    data = cv.transform([text]).toarray()
    output = model.predict(data)
    return output[0]

# Streamlit app
st.title("Stress Detection with Machine Learning")
st.write("Enter a text to determine if it indicates stress or not.")

user_input = st.text_area("Enter Text:")

if st.button("Predict"):
    if user_input:
        prediction = predict_stress(user_input)
        st.write(f"The prediction is: **{prediction}**")
    else:
        st.write("Please enter some text to analyze.")
