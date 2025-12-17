import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# TensorFlow import with error handling
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Prediksi Produktivitas - Kelompok 16",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎓 Prediksi Produktivitas Mahasiswa</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Learning - Kelompok 16 | Institut Teknologi Sumatera</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/ITERA_logo.svg/1200px-ITERA_logo.svg.png", width=100)
    st.title("📋 Menu")
    menu = st.radio("Pilih Menu:", ["🏠 Beranda", "🔮 Prediksi", "📊 Tentang Model", "👥 Tim Kami"])

@st.cache_data
def load_data():
    """Load and preprocess dataset"""
    try:
        data = pd.read_excel("Data Deep Learning Kel.16.xlsx")
        data = data.drop(0, axis=0).reset_index(drop=True)
        
        numeric_cols = [
            'Jumlah Jam Tidur',
            'Jumlah Jam Belajar',
            'Jumlah Sesi Belajar',
            'Lama Penggunaan Screen Time'
        ]
        
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        if 'Aktifitas Utama' in data.columns:
            le = LabelEncoder()
            data['Aktifitas Utama'] = le.fit_transform(data['Aktifitas Utama'])
        
        # Calculate Productivity
        data['Produktivitas'] = (
            0.2 * (data['Jumlah Jam Tidur'] / 8) +
            0.3 * (data['Jumlah Jam Belajar'] / data['Jumlah Jam Belajar'].max()) +
            0.2 * (data['Jumlah Sesi Belajar'] / data['Jumlah Sesi Belajar'].max()) +
            0.1 * (1 - data['Lama Penggunaan Screen Time'] / data['Lama Penggunaan Screen Time'].max())
        ) * 10
        
        data['Produktivitas'] = data['Produktivitas'].clip(1, 10)
        
        return data, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, False

@st.cache_resource
def train_model(data):
    """Train the MLP model"""
    if not TF_AVAILABLE:
        return None, None, None
    
    X = data[['Jumlah Jam Tidur', 'Jumlah Jam Belajar', 'Aktifitas Utama',
              'Lama Penggunaan Screen Time', 'Jumlah Sesi Belajar']]
    y = data['Produktivitas']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    with st.spinner("🔄 Melatih model... Mohon tunggu sebentar..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=10,
            verbose=0
        )
    
    return model, scaler, history

def predict_productivity(model, scaler, jam_tidur, jam_belajar, aktivitas, screen_time, sesi_belajar):
    """Make prediction"""
    input_data = np.array([[jam_tidur, jam_belajar, aktivitas, screen_time, sesi_belajar]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)
    return prediction[0][0]

def get_category(score):
    """Get productivity category"""
    if score < 4:
        return "Rendah", "🔴", "#ff4444"
    elif score < 7:
        return "Sedang", "🟡", "#ffbb33"
    else:
        return "Tinggi", "🟢", "#00C851"

# Main content based on menu selection
if menu == "🏠 Beranda":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Tujuan</h3>
            <p>Memprediksi tingkat produktivitas mahasiswa berdasarkan pola hidup sehari-hari</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Teknologi</h3>
            <p>Multi-Layer Perceptron (MLP) dengan TensorFlow/Keras</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data</h3>
            <p>Dataset survei mahasiswa dengan 5 fitur input</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📌 Fitur yang Dianalisis
    
    | No | Fitur | Deskripsi |
    |:--:|-------|-----------|
    | 1 | Jumlah Jam Tidur | Durasi tidur per hari (jam) |
    | 2 | Jumlah Jam Belajar | Durasi belajar per hari (jam) |
    | 3 | Jumlah Sesi Belajar | Frekuensi sesi belajar per hari |
    | 4 | Screen Time | Durasi penggunaan gadget (jam) |
    | 5 | Aktivitas Utama | Kategori aktivitas dominan |
    """)
    
    st.info("👈 Pilih menu **🔮 Prediksi** di sidebar untuk mulai memprediksi produktivitas Anda!")

elif menu == "🔮 Prediksi":
    st.markdown("---")
    st.subheader("🔮 Input Data untuk Prediksi")
    
    if not TF_AVAILABLE:
        st.error("⚠️ TensorFlow tidak terinstall. Silakan install dengan: `pip install tensorflow`")
    else:
        # Load data and train model
        data, data_loaded = load_data()
        
        if data_loaded and data is not None:
            model, scaler, history = train_model(data)
            
            if model is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    jam_tidur = st.slider("😴 Jumlah Jam Tidur", 4.0, 12.0, 7.0, 0.5)
                    jam_belajar = st.slider("📚 Jumlah Jam Belajar", 0.0, 15.0, 4.0, 0.5)
                    sesi_belajar = st.slider("📖 Jumlah Sesi Belajar", 1, 10, 3)
                
                with col2:
                    screen_time = st.slider("📱 Screen Time (jam)", 1.0, 12.0, 5.0, 0.5)
                    aktivitas = st.selectbox(
                        "🎯 Aktivitas Utama",
                        options=[0, 1, 2],
                        format_func=lambda x: ["Kuliah", "Organisasi", "Kerja Paruh Waktu"][x]
                    )
                
                st.markdown("---")
                
                if st.button("🎯 Prediksi Produktivitas", use_container_width=True, type="primary"):
                    prediction = predict_productivity(
                        model, scaler, jam_tidur, jam_belajar, aktivitas, screen_time, sesi_belajar
                    )
                    
                    category, emoji, color = get_category(prediction)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h2>Hasil Prediksi</h2>
                        <h1 style="font-size: 4rem;">{prediction:.2f}</h1>
                        <h3>{emoji} Tingkat Produktivitas: <span style="color: {color}">{category}</span></h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Rekomendasi")
                    if category == "Rendah":
                        st.warning("""
                        - Tingkatkan jam tidur hingga 7-8 jam per hari
                        - Kurangi screen time dan alokasikan untuk belajar
                        - Buat jadwal belajar yang teratur
                        """)
                    elif category == "Sedang":
                        st.info("""
                        - Pertahankan pola tidur yang baik
                        - Tingkatkan frekuensi sesi belajar
                        - Batasi penggunaan gadget untuk hiburan
                        """)
                    else:
                        st.success("""
                        - Pertahankan pola produktif Anda!
                        - Jaga keseimbangan antara belajar dan istirahat
                        - Terus tingkatkan kualitas belajar
                        """)

elif menu == "📊 Tentang Model":
    st.markdown("---")
    st.subheader("📊 Arsitektur Model MLP")
    
    st.markdown("""
    ### 🏗️ Struktur Neural Network
    
    ```
    Input Layer (5 neurons)
           ↓
    Hidden Layer 1 (32 neurons) + ReLU + Dropout(0.2)
           ↓
    Hidden Layer 2 (16 neurons) + ReLU + Dropout(0.2)
           ↓
    Hidden Layer 3 (8 neurons) + ReLU
           ↓
    Output Layer (1 neuron) + Linear
    ```
    
    ### ⚙️ Konfigurasi Training
    
    | Parameter | Nilai |
    |-----------|-------|
    | Optimizer | Adam |
    | Learning Rate | 0.001 |
    | Loss Function | Mean Squared Error (MSE) |
    | Batch Size | 10 |
    | Epochs | 50 |
    | Train/Test Split | 80/20 |
    
    ### 📈 Metrik Evaluasi
    
    - **MAE (Mean Absolute Error)**: Rata-rata kesalahan absolut prediksi
    - **RMSE (Root Mean Squared Error)**: Akar rata-rata kuadrat kesalahan
    - **R² Score**: Koefisien determinasi
    """)

elif menu == "👥 Tim Kami":
    st.markdown("---")
    st.subheader("👥 Kelompok 16 - Deep Learning")
    
    st.markdown("""
    ### 🏫 Institut Teknologi Sumatera
    **Program Studi Sains Data**
    
    ---
    
    ### 📚 Informasi Tugas
    
    - **Mata Kuliah**: Deep Learning
    - **Tugas**: Tugas Besar
    - **Topik**: Prediksi Produktivitas Mahasiswa dengan MLP
    
    ---
    
    ### 🔗 Repository
    
    GitHub: [Tugas-Besar_DeepLearning_Kel16](https://github.com/sains-data/Tugas-Besar_DeepLearning_Kel16)
    
    ---
    
    ### 📧 Kontak
    
    Untuk pertanyaan atau saran, silakan hubungi melalui GitHub Issues.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>© 2024 Kelompok 16 - Deep Learning | Institut Teknologi Sumatera</p>",
    unsafe_allow_html=True
)
