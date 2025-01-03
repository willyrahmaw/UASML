import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import joblib

# Load model, encoder, and scaler untuk buah
@st.cache_data
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def predict_fruit(model, input_data, scaler):
    if not hasattr(scaler, 'scale_'):
        raise ValueError("Scaler belum dilatih. Pastikan scaler telah di-fit sebelum digunakan.")
    
    # Debugging: Tampilkan data input sebelum scaling
    st.write("Input Data (sebelum scaling):", input_data)
    
    scaled_data = scaler.transform([input_data])  # Scaling input data
    
    # Debugging: Tampilkan data setelah scaling
    st.write("Scaled Data:", scaled_data)
    
    prediction = model.predict(scaled_data)
    
    # Debugging: Tampilkan hasil prediksi (encoded)
    st.write("Prediction (encoded):", prediction)
    
    return prediction

# Fungsi untuk prediksi pumpkin (neural network)
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

# Load model dan scaler untuk pumpkin
def load_neural_network_models():
    model_pumpkin = tf.keras.models.load_model('model.h5')  # Ganti dengan path model neural network Anda
    scaler_pumpkin = joblib.load('scaler.pkl')  # Ganti dengan path scaler Anda
    return model_pumpkin, scaler_pumpkin

# Pilihan aplikasi (buah atau pumpkin)
app_choice = st.radio("Pilih Jenis Prediksi", ("Prediksi Buah", "Prediksi Pumpkin"))

if app_choice == "Prediksi Buah":
    # Load model, scaler, dan encoder untuk prediksi buah
    model_fruit = load_model('model.sav')
    scaler_fruit = load_model('scaler.sav')
    encoder_fruit = load_model('encode.sav')

    # App title
    st.title("Aplikasi Prediksi Buah")
    st.subheader("Masukkan Input untuk Prediksi Buah")

    # Input user
    diameter = st.number_input("Diameter (cm)", min_value=0.0, step=0.1)
    weight = st.number_input("Weight (grams)", min_value=0.0, step=0.1)
    red = st.number_input("Red Value", min_value=0, max_value=255, step=1)
    green = st.number_input("Green Value", min_value=0, max_value=255, step=1)
    blue = st.number_input("Blue Value", min_value=0, max_value=255, step=1)

    # Susun input data ke dalam array
    input_data = np.array([diameter, weight, red, green, blue])

    # Tombol prediksi
    if st.button("Prediksi Buah"):
        try:
            prediction = predict_fruit(model_fruit, input_data, scaler_fruit)

            
            fruit_type = encoder_fruit.inverse_transform(prediction)

            st.success(f"Prediksi: {fruit_type[0]}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")

elif app_choice == "Prediksi Pumpkin":
    # Load model dan scaler untuk pumpkin
    model_pumpkin, scaler_pumpkin = load_neural_network_models()

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
