import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("🚢 Titanic Survival Predictor")

# User inputs
st.subheader("Enter passenger details:")

pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = Upper, 2 = Middle, 3 = Lower")
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)

# Convert inputs to match model format
sex_num = 1 if sex == "male" else 0
input_data = pd.DataFrame([[pclass, sex_num, age]], columns=["Pclass", "Sex", "Age"])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger would have **SURVIVED**.")
    else:
        st.error("💀 The passenger would have **NOT SURVIVED**.")

# Optional: Display model input
st.write("Model input:", input_data)
