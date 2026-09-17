import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load("Iris_model.pkl")

st.title("Machine Learning on Iris Dataset")

# Inputs for the flower characteristics
sepal_length = st.number_input("Sepal Length (cm)", value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", value=0.0)
petal_length = st.number_input("Petal Length (cm)", value=0.0)
petal_width = st.number_input("Petal Width (cm)", value=0.0)

if st.button("predict"):
    input_data = np.array([[sepal_length,
                            sepal_width,
                            petal_length,
                            petal_width]]).astype(np.float64)
    prediction = model.predict(input_data)
    st.success(f"Predicted species: {prediction[0]}")
