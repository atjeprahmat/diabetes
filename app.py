import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_xgb.pkl")

st.title("Prediksi Diabetes - XGBoost")

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 300)
bp = st.number_input("Blood Pressure", 0, 200)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0)
age = st.number_input("Age", 1, 120)

if st.button("Prediksi"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    pred = model.predict(data)[0]

    if pred == 1:
        st.error("Hasil: Berisiko Diabetes")
    else:
        st.success("Hasil: Tidak Berisiko Diabetes")
