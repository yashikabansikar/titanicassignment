import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title("🚢 Titanic Survival Prediction")

# Load trained model
titanic_model = pickle.load(open('titanic_model.pkl', 'rb'))

# User input fields
passenger_class = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
gender = st.selectbox("Sex", ["Male", "Female"])
passenger_age = st.number_input("Age", min_value=0, max_value=100, step=1)
siblings_spouses = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parents_children = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
ticket_fare = st.number_input("Fare", min_value=0.0, step=0.1)
embark_location = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert categorical inputs to match model's training format
gender_male = 1 if gender == "Male" else 0  # Model was trained with 'Sex_male'
embarked_from_Q = 1 if embark_location == "Q" else 0
embarked_from_S = 1 if embark_location == "S" else 0

# Organize user input in the correct format
input_data = {
    'Pclass': passenger_class,
    'Age': passenger_age,
    'SibSp': siblings_spouses,
    'Parch': parents_children,
    'Fare': ticket_fare,
    'Sex_male': gender_male,
    'Embarked_Q': embarked_from_Q,
    'Embarked_S': embarked_from_S
}

# Convert to DataFrame with correct structure
input_df = pd.DataFrame(input_data, index=[0])

# Predict button
if st.button("Predict Survival"):
    prediction_result = titanic_model.predict(input_df)

    # Display prediction outcome
    survival_status = "Survived 🟢" if prediction_result[0] == 1 else "Did Not Survive 🔴"
    st.success(f"Prediction: **{survival_status}**")
