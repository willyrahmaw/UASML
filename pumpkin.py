import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model dan scaler
@st.cache_data
def load_model(file_path):
    return tf.keras.models.load_model(file_path)  # Untuk model neural network

@st.cache_data
def load_scaler_and_encoder(scaler_path):
    scaler = joblib.load(scaler_path)  # Untuk scaler
    return scaler

# Fungsi untuk prediksi pumpkin
def predict_pumpkin(model, scaler, input_data):
    # Scaling data input sebelum prediksi
    input_data_scaled = scaler.transform([input_data])  # Pastikan input sudah sesuai dengan scaler

    # Prediksi kelas pumpkin
    prediction = model.predict(input_data_scaled)
    
    # Ambil indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Tentukan nama kelas berdasarkan indeks
    class_names = ["Çerçevelik", "Ürgüp Sivrisi"]  # Ganti dengan nama kelas pumpkin yang sesuai
    predicted_class = class_names[predicted_class_index]

    return predicted_class

# Load model dan scaler
model_pumpkin = load_model('model.h5')  # Ganti dengan path model neural network Anda
scaler_pumpkin = load_scaler_and_encoder('scaler.pkl')  # Ganti dengan path scaler Anda

# App title
st.title("Aplikasi Prediksi Pumpkin")
st.subheader("Masukkan Input untuk Prediksi Varietas Pumpkin")

# Input user
area = st.number_input("Area", min_value=0.0, step=0.1)
perimeter = st.number_input("Perimeter", min_value=0.0, step=0.1)
major_axis_length = st.number_input("Major Axis Length", min_value=0.0, step=0.1)
minor_axis_length = st.number_input("Minor Axis Length", min_value=0.0, step=0.1)
convex_area = st.number_input("Convex Area", min_value=0.0, step=0.1)
equiv_diameter = st.number_input("Equiv Diameter", min_value=0.0, step=0.1)
eccentricity = st.number_input("Eccentricity", min_value=0.0, step=0.1)
solidity = st.number_input("Solidity", min_value=0.0, step=0.1)
extent = st.number_input("Extent", min_value=0.0, step=0.1)
roundness = st.number_input("Roundness", min_value=0.0, step=0.1)
aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0, step=0.1)
compactness = st.number_input("Compactness", min_value=0.0, step=0.1)

# Susun input data ke dalam array
input_data = np.array([area, perimeter, major_axis_length, minor_axis_length, convex_area,
                       equiv_diameter, eccentricity, solidity, extent, roundness, aspect_ratio, compactness])

# Tombol prediksi
if st.button("Prediksi Varietas Pumpkin"):
    try:
        # Prediksi berdasarkan input
        predicted_pumpkin = predict_pumpkin(model_pumpkin, scaler_pumpkin, input_data)

        # Tampilkan hasil prediksi
        st.success(f"Prediksi Varietas: {predicted_pumpkin}")
    except Exception as e:
        # Tangani kesalahan
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
