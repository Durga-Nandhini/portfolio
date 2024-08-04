import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.header('Insurance Premium Prediction App')
st.write('Created by : Durga Nandhini M')

with open('lgb_model .pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Input fields
Age = st.slider("Select your age", 18, 65)
col1, col2, col3 = st.columns(3)
with col1:
    Diabetes = st.selectbox("Are you Diabetic?", ("Yes", "No"))
with col2:
    BloodPressureProblems = st.selectbox("High Blood Pressure?", ("Yes", "No"))
with col3:
    AnyChronicDiseases = st.selectbox("Any chronic diseases?", ("Yes", "No"))
col4, col5, col6 = st.columns(3)
with col4:
    KnownAllergies = st.selectbox("Are you Allergic?", ("Yes", "No"))
with col5:
    HistoryOfCancerInFamily = st.selectbox("History of Family Cancer?", ("Yes", "No"))
with col6:
    NumberOfMajorSurgeries = st.selectbox("No of surgeries undergone till date?", [0, 1, 2, 3])
col7, col8, col9 = st.columns(3)
with col7:
    AnyTransplants = st.selectbox("If you had a Transplant?", ("Yes", "No"))
with col8:
    Height = st.slider("Select your Height in CM", 145, 188)
with col9:
    Weight = st.slider("Select your Weight in KG", 51, 132)

encode_dict = {
    "Diabetes": {"Yes": 1, "No": 0},
    "BloodPressureProblems": {"Yes": 1, "No": 0},
    "AnyChronicDiseases": {"Yes": 1, "No": 0},
    "KnownAllergies": {"Yes": 1, "No": 0},
    "HistoryOfCancerInFamily": {"Yes": 1, "No": 0},
    "AnyTransplants": {"Yes": 1, "No": 0}
}

if st.button('Predict Premium Price'):
    # Encode the input features
    Diabetes = encode_dict["Diabetes"][Diabetes]
    BloodPressureProblems = encode_dict["BloodPressureProblems"][BloodPressureProblems]
    AnyChronicDiseases = encode_dict["AnyChronicDiseases"][AnyChronicDiseases]
    KnownAllergies = encode_dict["KnownAllergies"][KnownAllergies]
    HistoryOfCancerInFamily = encode_dict["HistoryOfCancerInFamily"][HistoryOfCancerInFamily]
    AnyTransplants = encode_dict["AnyTransplants"][AnyTransplants]

    # Create a DataFrame for consistent feature names
    X_train = pd.DataFrame({
        'Age': [Age],
        'Diabetes': [Diabetes],
        'BloodPressureProblems': [BloodPressureProblems],
        'AnyTransplants': [AnyTransplants],
        'AnyChronicDiseases': [AnyChronicDiseases],
        'Height': [Height],
        'Weight': [Weight],
        'KnownAllergies': [KnownAllergies],
        'HistoryOfCancerInFamily': [HistoryOfCancerInFamily],
        'NumberOfMajorSurgeries': [NumberOfMajorSurgeries]
    })

    predicted_price = loaded_model.predict(X_train)
    rounded_price = round(predicted_price[0], 2)  # Extract the scalar value and round it
    st.write(f'The predicted price is Rupees: {rounded_price}')
