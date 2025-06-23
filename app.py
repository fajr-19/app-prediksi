import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.neighbors import KNeighborsClassifier # Import a placeholder model if needed

# --- Placeholder: Train and save scaler/model if they don't exist ---
# In a real scenario, you would train your model and scaler on your dataset
# and save them beforehand. This is just an example to prevent FileNotFoundError.
import os
if not os.path.exists('knn_model.pkl') or not os.path.exists('scaler.pkl'):
    print("Training placeholder model and scaler...")
    # Create some dummy data to train a scaler
    dummy_data = pd.DataFrame({
        'Age': [20, 22, 25, 18, 21],
        'Sleep_Quality': [5, 7, 6, 4, 8],
        'Depression_Score': [3, 6, 4, 8, 2],
        'Anxiety_Score': [4, 5, 7, 9, 1],
        'Financial_Stress': [6, 3, 8, 7, 2]
    })
    # Fit and save the scaler
    scaler_placeholder = StandardScaler()
    scaler_placeholder.fit(dummy_data)
    joblib.dump(scaler_placeholder, 'scaler.pkl')
    print("Placeholder scaler saved!")

    # Create and save a placeholder model (replace with your actual model training)
    # Dummy target variable for the placeholder KNN model
    dummy_target = [0, 1, 0, 2, 0] # Example stress levels (0: Rendah, 1: Sedang, 2: Tinggi)
    knn_placeholder = KNeighborsClassifier(n_neighbors=3)
    knn_placeholder.fit(scaler_placeholder.transform(dummy_data), dummy_target)
    joblib.dump(knn_placeholder, 'knn_model.pkl')
    print("Placeholder model saved!")

# --- End Placeholder ---


# Load model & scaler
# Assuming 'knn_model.pkl' and 'scaler.pkl' exist from a previous run
try:
    knn = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model = knn # Assuming your model variable is named 'model' for prediction
    print("Model & scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files (knn_model.pkl or scaler.pkl) not found.")
    st.stop() # Stop the Streamlit app if files are missing

st.title("Prediksi Tingkat Stres Mahasiswa")

# Input
usia = st.slider("Usia", 17, 30, 21)
tidur = st.slider("Rata-rata jam tidur", 0, 10, 6)
depresi = st.slider("Skor depresi", 0, 10, 5)
cemas = st.slider("Skor kecemasan", 0, 10, 5)
stres_uang = st.slider("Stres finansial", 0, 10, 5)

# Prediksi
data = pd.DataFrame([[usia, tidur, depresi, cemas, stres_uang]],
                    columns=['Age', 'Sleep_Quality', 'Depression_Score', 'Anxiety_Score', 'Financial_Stress'])

# Ensure scaler and model are loaded or defined before this point
data_scaled = scaler.transform(data)
pred = model.predict(data_scaled)[0]

kategori = ['Rendah', 'Sedang', 'Tinggi']
st.success(f"Tingkat Stres Anda: **{kategori[int(pred)]}**")

# Note: Remember to run this script using 'streamlit run your_script_name.py'
# to see the Streamlit app. Running it directly in a Jupyter cell will not
# display the interactive app.