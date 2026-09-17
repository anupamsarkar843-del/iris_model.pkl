import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Page title
st.title("Machine Learning on Iris Dataset")

# Input labels
sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

# Prediction button
if st.button("Predict"):    
    # Create a numpy array from the input values
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Predict the species
    prediction = model.predict(input_features)
    st.success(f"The predicted species is: {prediction[0]}")
