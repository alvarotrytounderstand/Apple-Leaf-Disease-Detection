# 🍎 Apple Leaf Disease Detection (Multi-label)

Aplikasi berbasis web untuk mendeteksi dan mengklasifikasikan penyakit pada daun apel menggunakan arsitektur **EfficientNetB3**. Sistem ini mampu mendeteksi infeksi ganda (*Multiple Diseases*) secara bersamaan dan dilengkapi dengan fitur **Grad-CAM** (*Explainable AI*) untuk memvisualisasikan area fokus penyakit pada daun berupa peta panas (*heatmap*).

---

## ✨ Fitur Utama

- **Multi-label Classification:** Mampu mendeteksi lebih dari satu penyakit dalam satu daun (Sehat, Karat/Rust, Keropeng/Scab, dan Infeksi Ganda).
- **Explainable AI (Grad-CAM):** Menampilkan *heatmap* untuk membuktikan transparansi keputusan model.
- **Interaktif:** Antarmuka yang mudah digunakan berbasis web.

---

## 🛠️ Teknologi yang Digunakan

| Kategori | Teknologi |
|---|---|
| **Bahasa** | Python 3.9+ |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Web Framework** | Streamlit |
| **Computer Vision** | OpenCV, Pillow, NumPy |

---

## 🚀 Cara Instalasi dan Menjalankan Aplikasi

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di komputer lokal.

### 1. Clone Repository

Buka terminal/command prompt, lalu jalankan perintah berikut:

```bash
git clone https://github.com/alvarotrytounderstand/Apple-Leaf-Disease-Detection.git
cd Apple-Leaf-Disease-Detection
```

---

### 2. Buat Virtual Environment *(Sangat Disarankan)*

Agar library tidak bentrok dengan project lain, buat dan aktifkan virtual environment:

**Windows:**
```bash
python -m venv env
.\env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

---

### 3. Install Library yang Dibutuhkan

Install semua library (TensorFlow, Streamlit, OpenCV, dll.) dengan satu perintah:

```bash
pip install -r requirements.txt
```

---

### 4. Download Model AI ⚠️ PENTING!

File model AI tidak disertakan di GitHub karena ukurannya yang besar. Ikuti langkah berikut:

1. Buat folder baru bernama **`models`** di dalam folder project ini.
2. Download file model **`efficientnetb3_multilabel_finetuned_best.keras`** melalui link Google Drive berikut:
   > 👉 **[Download Model AI di Sini](#)**
3. Pindahkan file model yang sudah didownload ke dalam folder **`models/`** yang baru saja dibuat.

Struktur folder seharusnya terlihat seperti ini:
```
Apple-Leaf-Disease-Detection/
├── models/
│   └── efficientnetb3_multilabel_finetuned_best.keras
├── app.py
├── requirements.txt
└── ...
```

---

### 5. Jalankan Aplikasi

Jika semua langkah di atas sudah selesai, jalankan server Streamlit dengan perintah:

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser kamu (biasanya di `http://localhost:8501`).

**Selamat mencoba! 🍏🔍**

---

## 📁 Struktur Project

```
Apple-Leaf-Disease-Detection/
├── models/                  # Folder untuk menyimpan file model (.keras)
├── app.py                   # File utama aplikasi Streamlit
├── requirements.txt         # Daftar library yang dibutuhkan
└── README.md                # Dokumentasi project
```

---

## 📄 Lisensi

Project ini dibuat untuk keperluan edukasi dan penelitian.