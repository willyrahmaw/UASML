import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model dan encoder
@st.cache_data
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def predict_fruit(model, input_data):
    # Debugging: Tampilkan data input
    st.write("Input Data:", input_data)

    prediction = model.predict([input_data])  # Prediksi langsung tanpa scaling

    # Debugging: Tampilkan hasil prediksi (encoded)
    st.write("Prediction (encoded):", prediction)

    return prediction

# Load model dan encoder
model_fruit = load_model('model.sav')
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
        # Prediksi berdasarkan input
        prediction = predict_fruit(model_fruit, input_data)

        # Dekode hasil prediksi menjadi label
        fruit_type = encoder_fruit.inverse_transform(prediction)

        # Tampilkan hasil prediksi
        st.success(f"Prediksi: {fruit_type[0]}")
    except Exception as e:
        # Tangani kesalahan
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")