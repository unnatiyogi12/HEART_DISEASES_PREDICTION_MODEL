import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNeighborsClassifier_heart_diseases.pkl')
scaler = joblib.load('scaler.pkl')
exprected_columns = joblib.load('columns.pkl')

st.title('Heart Stroke Prediction App')
st.markdown("Provides the following details")

age = st.slider('Age', 18, 100, 40)
sex = st.selectbox("SEX",["Male", "Female"])
ChestPainType = st.selectbox("Chest Pain Type",["ATA", "NAP", "TA", "ASY"])
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl",["Yes", "No"])
restingECG = st.selectbox("Resting ECG",["Normal", "ST", "LVH"])
maxHR = st.slider("Max Heart Rate", 60, 220, 150)
exerciseAngina = st.selectbox("Exercise Induced Angina",["Yes", "No"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
ST_Slope = st.selectbox("ST Slope",["Up", "Flat", "Down"])

if st.button('predict'):
    raw_data = {
        'Age': age, 
        'RestingBP': restingBP,
        'Cholesterol': cholesterol, 
        'FastingBS': 1 if fastingBS == "Yes" else 0,
        'MaxHR': maxHR,
        'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == "Male" else 0,
        'ChestPainType_ATA': 1 if ChestPainType == "ATA" else 0,
        'ChestPainType_NAP': 1 if ChestPainType == "NAP" else 0,
        'ChestPainType_TA': 1 if ChestPainType == "TA" else 0,
        'RestingECG_Normal': 1 if restingECG == "Normal" else 0,  
        'RestingECG_ST': 1 if restingECG == "ST" else 0,
        'ExerciseAngina_Y': 1 if exerciseAngina == "Yes" else 0,
        'ST_Slope_Flat': 1 if ST_Slope == "Flat" else 0,
        'ST_Slope_Up': 1 if ST_Slope == "Up" else 0
    }

    input_data = pd.DataFrame([raw_data])
    # agar koi column exsisi nhi krta column.pkl m to default 0 value krdo
    for col in exprected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[exprected_columns]
    # agr use kr rhe model like KNN, SVM, Logistic Regression to scale the data
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)[0]
    if prediction == 1:
        st.error(" ⚠️ The patient is likely to have a heart stroke.")
    else:
        st.success("🪄 The patient is unlikely to have a heart stroke.")
