import streamlit as st
import pandas as pd
import pickle

with open("diabetes_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
columns = model_data["columns"]

st.title("Diabetes Risk Predictor")
st.write("Enter the following medical information to estimate your diabetes risk:")


pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=500, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=10, max_value=100, value=30)

user_input = {
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
}

input_df = pd.DataFrame(user_input)[columns]
scaled_input = scaler.transform(input_df)


if st.button("Predict"):
    probability = model.predict_proba(scaled_input)[0][1]
    risk_percent = round(probability * 100, 2)

    st.markdown(f"### Estimated Risk: {risk_percent}%")
    
    if risk_percent >= 50:
        st.error("You show a high risk of diabetes. Please consult a doctor.")
    else:
        st.success("Your risk of diabetes is low. Keep maintaining a healthy lifestyle.")