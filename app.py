import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_xgb.pkl")

st.title("Prediksi Risiko Diabetes Menggunakan XGBoost")

st.markdown("""
Aplikasi ini menggunakan model XGBoost untuk memprediksi risiko diabetes berdasarkan data kesehatan Anda.
Masukkan nilai-nilai berikut untuk mendapatkan hasil prediksi.
""")

st.header("Masukkan Data Kesehatan")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Jumlah Kehamilan", 0, 20, help="Jumlah kehamilan yang pernah dialami (0-20)")
    glucose = st.number_input("Kadar Glukosa", 0, 300, help="Kadar glukosa darah dalam mg/dL (0-300)")
    bp = st.number_input("Tekanan Darah", 0, 200, help="Tekanan darah sistolik dalam mmHg (0-200)")
    skin = st.number_input("Ketebalan Kulit", 0, 100, help="Ketebalan lipatan kulit trisep dalam mm (0-100)")

with col2:
    insulin = st.number_input("Kadar Insulin", 0, 900, help="Kadar insulin serum dalam mu U/ml (0-900)")
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", 0.0, 70.0, help="Indeks massa tubuh (0.0-70.0)")
    dpf = st.number_input("Fungsi Pedigree Diabetes", 0.0, 5.0, help="Fungsi pedigree diabetes (0.0-5.0)")
    age = st.number_input("Usia", 1, 120, help="Usia dalam tahun (1-120)")

st.header("Hasil Prediksi")

if st.button("Lakukan Prediksi"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    pred = model.predict(data)[0]
    print(pred)

    if pred == 1:
        st.error("Hasil Prediksi: Anda Berisiko Terkena Diabetes")
        st.markdown("""
        **Rekomendasi:**
        - Konsultasikan dengan dokter untuk pemeriksaan lebih lanjut
        - Lakukan pola hidup sehat: olahraga teratur, diet seimbang
        - Pantau kadar gula darah secara rutin
        """)
    else:
        st.success("Hasil Prediksi: Anda Tidak Berisiko Terkena Diabetes", icon="✅")
        st.markdown("""
        **Tetap Jaga Kesehatan:**
        - Lanjutkan pola hidup sehat
        - Lakukan pemeriksaan kesehatan rutin
        - Jaga berat badan ideal dan aktif bergerak
        """)
