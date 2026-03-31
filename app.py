import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Pendeteksi Penyakit Daun Apel", page_icon="🍎", layout="wide")

st.title("🍎 Sistem Deteksi Multi-label Penyakit Daun Apel")
st.markdown("**Model:** EfficientNetB3 | **Fitur:** Multi-label Classification & Grad-CAM (Explainable AI)")
st.markdown("---")

# 2. Load Model (Cache biar gak loading terus tiap upload gambar)
@st.cache_resource
def load_trained_model():
    model_path = 'models/efficientnetb3_multilabel_finetuned_best.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"❌ File model tidak ditemukan di {model_path}. Pastikan path benar!")
        return None

model = load_trained_model()

# ==========================================
# MESIN GRAD-CAM (SENJATA PAMUNGKAS)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Buat model yang memetakan input ke output layer konvolusi terakhir dan prediksi akhir
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Hitung gradien dari output terhadap feature map konvolusi terakhir
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Kalikan setiap channel di feature map dengan "kepentingannya" (pooled_grads)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisasi heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    # Ubah heatmap ke warna RGB
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap agar ukurannya sama dengan gambar asli
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    # Gabungkan gambar asli dengan heatmap
    superimposed_img = jet * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# ==========================================
# ANTARMUKA UTAMA
# ==========================================
uploaded_file = st.file_uploader("📂 Upload foto daun apel di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    # Buka gambar asli
    img_pil = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("📷 Citra Asli")
        st.image(img_pil, use_container_width=True)
    
    # Preprocessing yang BENAR (Anti Dosa 255)
    with st.spinner('Sedang dianalisis oleh Model Dewa...'):
        img_resized = img_pil.resize((300, 300))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array_preprocessed = preprocess_input(img_array) # 🔥 INI YANG BENER
        
        # Eksekusi Prediksi
        predictions = model.predict(img_array_preprocessed)[0]
        
    # Tampilkan Hasil Prediksi Multi-label
    st.markdown("### 📊 Hasil Analisis Penyakit (Probabilitas Independen):")
    classes = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']
    
    for i, class_name in enumerate(classes):
        prob = predictions[i]
        # Bikin warna progress bar beda kalau di atas 50%
        bar_color = "green" if class_name == 'Healthy' and prob > 0.5 else ("red" if prob > 0.5 else "gray")
        
        st.write(f"**{class_name}**")
        st.progress(float(prob))
        st.write(f"{prob*100:.2f}%")

    # Ambil index penyakit dengan probabilitas tertinggi untuk Grad-CAM
    main_idx = np.argmax(predictions)
    
    # Generate & Tampilkan Grad-CAM
    with col2:
        st.subheader("🔥 Fokus AI (Grad-CAM Heatmap)")
        try:
            # Note: 'top_activation' adalah nama layer conv terakhir di EfficientNetB3
            # Jika error, kita perlu cek nama layer di model lu, tapi biasanya ini aman.
            heatmap = make_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name='top_activation', pred_index=main_idx)
            
            # Kembalikan gambar asli ke array numpy untuk digabung dengan heatmap
            img_original_array = np.array(img_pil.resize((300, 300)))
            heatmap_overlay = overlay_heatmap(img_original_array, heatmap)
            
            st.image(heatmap_overlay, use_container_width=True, caption=f"Area fokus AI untuk deteksi: {classes[main_idx]}")
        except Exception as e:
            st.warning("Heatmap tidak dapat ditampilkan karena struktur model terbungkus berbeda. Tapi prediksi tetap aman!")
            # st.write(e) # Buka comment ini kalau mau liat errornya

    st.markdown("---")
    # Kesimpulan Akhir
    if predictions[main_idx] > 0.5:
        st.success(f"**KESIMPULAN:** Daun terdeteksi kuat mengalami gejala **{classes[main_idx]}**.")
    else:
        st.warning("**PERINGATAN:** Model kurang yakin (<50%). Pastikan pencahayaan foto cukup dan objek daun terlihat jelas.")