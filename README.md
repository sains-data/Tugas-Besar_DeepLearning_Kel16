# 🎓 Tugas Besar Deep Learning - Kelompok 16

## Prediksi Produktivitas Mahasiswa dengan Multi-Layer Perceptron (MLP)

### 📋 Deskripsi Proyek
Aplikasi web berbasis Streamlit untuk memprediksi tingkat produktivitas mahasiswa menggunakan model Deep Learning (MLP). Model ini menganalisis faktor-faktor seperti jam tidur, jam belajar, sesi belajar, dan penggunaan screen time.

### 🏗️ Arsitektur Model
- **Input Layer**: 5 fitur
- **Hidden Layer 1**: 32 neuron + ReLU + Dropout 0.2
- **Hidden Layer 2**: 16 neuron + ReLU + Dropout 0.2
- **Hidden Layer 3**: 8 neuron + ReLU
- **Output Layer**: 1 neuron (Linear)

### 📊 Fitur Input
| Fitur | Deskripsi | Range |
|-------|-----------|-------|
| Jumlah Jam Tidur | Durasi tidur per hari | 4-12 jam |
| Jumlah Jam Belajar | Durasi belajar per hari | 0-15 jam |
| Jumlah Sesi Belajar | Frekuensi sesi belajar | 1-10 sesi |
| Screen Time | Penggunaan gadget | 1-12 jam |
| Aktivitas Utama | Kategori aktivitas | Kuliah/Organisasi/Kerja |

### 🚀 Cara Menjalankan

#### 1. Clone Repository
```bash
git clone https://github.com/sains-data/Tugas-Besar_DeepLearning_Kel16.git
cd Tugas-Besar_DeepLearning_Kel16
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

### 📈 Evaluasi Model
- **MAE**: ~0.73
- **RMSE**: ~0.94
- **R² Score**: ~0.37

### 👥 Anggota Kelompok 16
| No | Nama | NIM |
|:--:|------|-----|
| 1 | Chalifia Wananda | 122450076 |
| 2 | Rendra Eka Prayoga | 122450112 |
| 3 | Elisabeth Claudia Simanjuntak | 122450123 |

**Program Studi**: Sains Data  
**Institusi**: Institut Teknologi Sumatera

### 📁 Struktur File
```
├── app.py                          # Aplikasi Streamlit
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi
├── Data Deep Learning Kel.16.xlsx  # Dataset
└── tubes_dl.ipynb                  # Jupyter Notebook
```

### 📄 Lisensi
Proyek ini dibuat untuk keperluan akademik - Tugas Besar Deep Learning.
