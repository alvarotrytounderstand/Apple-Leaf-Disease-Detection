import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import io
import json
import time
from datetime import datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="LeafScan — Deteksi Penyakit Daun Apel",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:          #f7faf5;
    --bg-card:     #ffffff;
    --bg-soft:     #eef5eb;
    --border:      #d4e6cc;
    --green-dark:  #2d6a4f;
    --green-mid:   #52b788;
    --green-light: #b7e4c7;
    --mint:        #95d5b2;
    --text-main:   #1b3a2d;
    --text-muted:  #6b8f71;
    --text-dim:    #a8c5a0;
    --healthy:     #52b788;
    --rust:        #bc6c25;
    --scab:        #a68a5b;
    --multi:       #774936;
    --warn:        #dda15e;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text-main) !important;
}
.stApp { background-color: var(--bg) !important; }
.block-container { padding: 2.5rem 3rem 5rem !important; max-width: 1300px !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--green-light); border-radius: 10px; }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #d8f3dc 0%, #f0faf2 60%, #ffffff 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🌿';
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.12;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--green-mid);
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.hero-title {
    font-family: 'Lora', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--green-dark);
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-title em {
    font-style: italic;
    color: var(--green-mid);
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 0;
    max-width: 520px;
    line-height: 1.6;
}
.hero-pills {
    display: flex;
    gap: 0.5rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.pill {
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 500;
    letter-spacing: 0.3px;
}
.pill-green { background: #d8f3dc; color: var(--green-dark); }
.pill-mint  { background: #b7e4c7; color: #1b4332; }
.pill-gray  { background: #f0f0f0; color: #666; }

/* ── SECTION HEADER ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1.5px solid var(--border);
}
.sec-leaf { font-size: 1rem; }
.sec-title {
    font-family: 'Lora', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--green-dark);
}

/* ── METRIC CARDS ── */
.metric-card {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.metric-card.active {
    border-color: var(--card-color);
    box-shadow: 0 4px 20px var(--card-shadow);
}
.metric-card.active::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-color);
    border-radius: 14px 14px 0 0;
}
.metric-icon { font-size: 1.2rem; margin-bottom: 0.4rem; }
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
    font-weight: 500;
}
.metric-value {
    font-family: 'Lora', serif;
    font-size: 1.8rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-bar-wrap {
    height: 4px;
    background: var(--bg-soft);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.metric-bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.metric-status { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.4rem; }

/* ── IMG CARD HEADER ── */
.img-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    font-weight: 500;
    padding: 0.6rem 0;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── VERDICT ── */
.verdict-card {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.5rem;
    border: 1.5px solid;
}
.verdict-card.healthy { background: #d8f3dc; border-color: #95d5b2; }
.verdict-card.disease { background: #fff8f0; border-color: #dda15e; }
.verdict-card.unsure  { background: #fafafa; border-color: #e0e0e0; }
.verdict-eyebrow {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.verdict-main {
    font-family: 'Lora', serif;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.verdict-sub { font-size: 0.88rem; color: var(--text-muted); line-height: 1.5; }
.verdict-action {
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    background: rgba(255,255,255,0.6);
    border-radius: 10px;
    font-size: 0.84rem;
    color: var(--text-main);
    border-left: 3px solid;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid var(--border) !important;
}
.sb-logo {
    font-family: 'Lora', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--green-dark);
    margin-bottom: 0.15rem;
}
.sb-version { font-size: 0.7rem; color: var(--text-dim); margin-bottom: 1.5rem; }
.sb-section {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-dim);
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
    font-weight: 500;
}
.sb-status {
    font-size: 0.75rem;
    color: var(--text-muted);
    line-height: 2;
    background: var(--bg-soft);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}
.online-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green-mid);
    margin-right: 5px;
    animation: blink 2.5s ease-in-out infinite;
    vertical-align: middle;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── LOADING ── */
.loading-wrap {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.loading-title {
    font-family: 'Lora', serif;
    font-size: 1rem;
    color: var(--green-dark);
    margin-bottom: 0.8rem;
}
.loading-bar-wrap {
    height: 4px;
    background: var(--bg-soft);
    border-radius: 4px;
    overflow: hidden;
}
.loading-bar {
    height: 100%;
    width: 50%;
    background: linear-gradient(90deg, var(--green-mid), var(--mint));
    border-radius: 4px;
    animation: sweep 1.4s ease-in-out infinite;
}
@keyframes sweep { 0%{margin-left:-50%} 100%{margin-left:100%} }
.loading-hint { font-size: 0.75rem; color: var(--text-dim); margin-top: 0.7rem; }

/* ── DOWNLOAD BTN ── */
.stDownloadButton button {
    background: var(--bg-soft) !important;
    color: var(--green-dark) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stDownloadButton button:hover {
    background: var(--green-light) !important;
    border-color: var(--green-mid) !important;
}

/* ── EMPTY STATE ── */
.empty-state {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Lora', serif;
    font-size: 1.3rem;
    color: var(--green-dark);
    margin-bottom: 0.4rem;
}
.empty-sub { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.2rem; line-height: 1.6; }

/* ── MISC ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ──
CLASSES = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']
CLASS_ICONS  = { 'Healthy': '🌿', 'Multiple Diseases': '⚠️', 'Rust': '🍂', 'Scab': '🌰' }
CLASS_COLORS = { 'Healthy': '#52b788', 'Multiple Diseases': '#774936', 'Rust': '#bc6c25', 'Scab': '#a68a5b' }
CLASS_SHADOW = { 'Healthy': '#52b78822', 'Multiple Diseases': '#77493622', 'Rust': '#bc6c2522', 'Scab': '#a68a5b22' }
CLASS_DESC   = {
    'Healthy':           'Daun dalam kondisi sehat. Tidak ditemukan tanda-tanda infeksi atau penyakit.',
    'Multiple Diseases': 'Terdeteksi lebih dari satu jenis penyakit secara bersamaan pada daun ini.',
    'Rust':              'Penyakit karat daun (Cedar Apple Rust) akibat jamur Gymnosporangium juniperi-virginianae.',
    'Scab':              'Penyakit kudis apel (Apple Scab) akibat jamur Venturia inaequalis.',
}
CLASS_ACTION = {
    'Healthy':           '✓ Tidak diperlukan tindakan khusus. Lanjutkan pemantauan rutin.',
    'Multiple Diseases': '⚠ Segera konsultasikan dengan ahli agronomi atau penyuluh pertanian.',
    'Rust':              '⚠ Semprotkan fungisida berbahan aktif myclobutanil atau mancozeb secara merata.',
    'Scab':              '⚠ Pangkas dan musnahkan daun terinfeksi, aplikasikan captan atau thiram.',
}

# ── SIDEBAR ──
with st.sidebar:
    st.markdown('<div class="sb-logo">🌿 LeafScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-version">Sistem Deteksi Penyakit Daun Apel · v2.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Konfigurasi Analisis</div>', unsafe_allow_html=True)
    threshold = st.slider("Ambang Batas Kepercayaan (%)", 10, 90, 50, 5,
                          help="Prediksi di atas nilai ini dianggap positif terdeteksi")
    st.caption(f"Label aktif pada probabilitas ≥ **{threshold}%**")

    gradcam_layer = st.selectbox("Grad-CAM Layer",
        ["top_activation", "block7a_expand_activation", "block6a_expand_activation"],
        help="Layer konvolusi untuk menghasilkan heatmap")

    st.markdown('<div class="sb-section">Tampilan</div>', unsafe_allow_html=True)
    show_all_gradcam = st.toggle("Multi Grad-CAM per label", value=False)
    show_raw_scores  = st.toggle("Tampilkan nilai mentah", value=False)

    st.markdown('<div class="sb-section">Status Sistem</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-status">
        <span class="online-dot"></span> Model aktif<br>
        Arsitektur · EfficientNetB3<br>
        Resolusi input · 300 × 300 px<br>
        Jumlah kelas · {len(CLASSES)}<br>
        Ambang batas · {threshold}%
    </div>""", unsafe_allow_html=True)

# ── MODEL ──
@st.cache_resource
def load_trained_model():
    model_path = 'models/efficientnetb3_multilabel_finetuned_best.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    st.error(f"❌ Model tidak ditemukan di `{model_path}`")
    return None

model = load_trained_model()

# ── GRAD-CAM ──
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.42):
    h = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(h, cv2.COLORMAP_YlGn)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    out = jet * alpha + img * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def build_report(predictions, filename, threshold):
    detected = [CLASSES[i] for i, p in enumerate(predictions) if p * 100 >= threshold]
    return json.dumps({
        "timestamp": datetime.now().isoformat(),
        "file": filename,
        "threshold_pct": threshold,
        "predictions": {cls: f"{predictions[i]*100:.2f}%" for i, cls in enumerate(CLASSES)},
        "terdeteksi": detected if detected else ["Tidak ada deteksi di atas threshold"],
        "diagnosa_utama": CLASSES[np.argmax(predictions)],
        "kepercayaan": f"{predictions[np.argmax(predictions)]*100:.2f}%",
        "rekomendasi": CLASS_ACTION[CLASSES[np.argmax(predictions)]]
    }, indent=2, ensure_ascii=False)

# ── HERO ──
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🌱 Sistem Deteksi Penyakit Tanaman</div>
    <div class="hero-title">Analisis Kesehatan<br><em>Daun Apel</em></div>
    <div class="hero-sub">
        Upload foto daun apel dan sistem akan mengidentifikasi kondisi kesehatan
        menggunakan model deep learning EfficientNetB3 dengan visualisasi Grad-CAM.
    </div>
    <div class="hero-pills">
        <span class="pill pill-green">EfficientNetB3</span>
        <span class="pill pill-mint">Grad-CAM</span>
        <span class="pill pill-gray">Multi-label · 4 Kelas</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT GAMBAR (TAB UPLOAD & KAMERA) ──
tab1, tab2 = st.tabs(["📂 Unggah Gambar", "📸 Buka Kamera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Unggah foto daun apel",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with tab2:
    camera_file = st.camera_input("Jepret foto daun langsung dari kamera")

# Logika pinter milih sumber gambar
image_source = camera_file if camera_file is not None else uploaded_file

# ── MAIN ──
if image_source is not None and model is not None:
    img_pil = Image.open(image_source).convert('RGB')

# ── MAIN ──
if uploaded_file is not None and model is not None:
    img_pil = Image.open(uploaded_file).convert('RGB')

    # Loading
    ph = st.empty()
    ph.markdown("""
    <div class="loading-wrap">
        <div class="loading-title">🔍 Sedang menganalisis daun...</div>
        <div class="loading-bar-wrap"><div class="loading-bar"></div></div>
        <div class="loading-hint">Preprocessing gambar · Menjalankan model · Membuat heatmap</div>
    </div>""", unsafe_allow_html=True)

    img_resized      = img_pil.resize((300, 300))
    img_array        = image.img_to_array(img_resized)
    img_expanded     = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_expanded)
    predictions      = model.predict(img_preprocessed, verbose=0)[0]
    time.sleep(0.3)
    ph.empty()

    main_idx        = int(np.argmax(predictions))
    main_class      = CLASSES[main_idx]
    main_prob       = predictions[main_idx]
    detected_labels = [CLASSES[i] for i, p in enumerate(predictions) if p * 100 >= threshold]
    img_np          = np.array(img_pil.resize((300, 300)))

    # ── METRIC CARDS ──
    st.markdown("""
    <div class="sec-header">
        <span class="sec-leaf">📊</span>
        <span class="sec-title">Hasil Analisis — Probabilitas Per Kelas</span>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, cls in enumerate(CLASSES):
        pct       = predictions[i] * 100
        color     = CLASS_COLORS[cls]
        shadow    = CLASS_SHADOW[cls]
        icon      = CLASS_ICONS[cls]
        is_active = pct >= threshold
        active    = "active" if is_active else ""
        bar_color = color if is_active else "#d4e6cc"
        val_color = color if is_active else "#a8c5a0"
        status    = f"✓ Terdeteksi" if is_active else "— Tidak terdeteksi"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card {active}" style="--card-color:{color};--card-shadow:{shadow}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{cls}</div>
                <div class="metric-value" style="color:{val_color}">{pct:.1f}<span style="font-size:1rem;font-weight:400">%</span></div>
                <div class="metric-bar-wrap">
                    <div class="metric-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
                </div>
                <div class="metric-status">{status}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── IMAGES ──
    st.markdown("""
    <div class="sec-header">
        <span class="sec-leaf">🔬</span>
        <span class="sec-title">Citra Asli & Visualisasi Grad-CAM</span>
    </div>""", unsafe_allow_html=True)

    if show_all_gradcam and len(detected_labels) > 1:
        gcols = st.columns(len(detected_labels) + 1)
        with gcols[0]:
            st.markdown('<div class="img-label">📷 Foto asli</div>', unsafe_allow_html=True)
            st.image(img_pil, use_container_width=True)
        for j, label in enumerate(detected_labels):
            idx   = CLASSES.index(label)
            color = CLASS_COLORS[label]
            try:
                hm  = make_gradcam_heatmap(img_preprocessed, model, gradcam_layer, pred_index=idx)
                ov  = overlay_heatmap(img_np, hm)
                with gcols[j + 1]:
                    st.markdown(f'<div class="img-label">{CLASS_ICONS[label]} {label}</div>', unsafe_allow_html=True)
                    st.image(ov, use_container_width=True)
            except Exception:
                pass
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="img-label">📷 Foto asli</div>', unsafe_allow_html=True)
            st.image(img_pil, use_container_width=True)
        with col2:
            st.markdown(f'<div class="img-label">{CLASS_ICONS[main_class]} Grad-CAM · {main_class}</div>', unsafe_allow_html=True)
            try:
                hm = make_gradcam_heatmap(img_preprocessed, model, gradcam_layer, pred_index=main_idx)
                ov = overlay_heatmap(img_np, hm)
                st.image(ov, use_container_width=True, caption=f"Area fokus model untuk kelas: {main_class}")
            except Exception:
                st.warning("Grad-CAM tidak tersedia untuk konfigurasi model ini.")

    if show_raw_scores:
        with st.expander("Nilai prediksi mentah"):
            for i, cls in enumerate(CLASSES):
                st.code(f"{cls:20s}: {predictions[i]:.6f}  ({predictions[i]*100:.2f}%)", language=None)

    # ── VERDICT ──
    if main_prob >= threshold / 100:
        if main_class == 'Healthy':
            vclass, vcolor = "healthy", "#2d6a4f"
        else:
            vclass, vcolor = "disease", "#bc6c25"
        icon  = CLASS_ICONS[main_class]
        color = CLASS_COLORS[main_class]
        st.markdown(f"""
        <div class="verdict-card {vclass}">
            <div class="verdict-eyebrow">Hasil Diagnosa</div>
            <div class="verdict-main" style="color:{vcolor}">{icon} {main_class} — {main_prob*100:.1f}% kepercayaan</div>
            <div class="verdict-sub">{CLASS_DESC[main_class]}</div>
            <div class="verdict-action" style="border-color:{color}">{CLASS_ACTION[main_class]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-card unsure">
            <div class="verdict-eyebrow">Hasil Diagnosa</div>
            <div class="verdict-main" style="color:#a68a5b">⚡ Kepercayaan Rendah — {main_prob*100:.1f}%</div>
            <div class="verdict-sub">Model tidak cukup yakin dengan hasil ini. Pastikan foto diambil dengan pencahayaan yang cukup, objek daun terlihat jelas, dan tidak buram.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DOWNLOAD ──
    st.markdown("""
    <div class="sec-header">
        <span class="sec-leaf">📥</span>
        <span class="sec-title">Unduh Hasil</span>
    </div>""", unsafe_allow_html=True)

    d1, d2, _ = st.columns([1, 1, 2])
    with d1:
        st.download_button(
            "⬇ Laporan (JSON)",
            data=build_report(predictions, image_source.name, threshold),
            file_name=f"leafscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    with d2:
        try:
            hm  = make_gradcam_heatmap(img_preprocessed, model, gradcam_layer, pred_index=main_idx)
            ov  = overlay_heatmap(img_np, hm)
            buf = io.BytesIO()
            Image.fromarray(ov).save(buf, format="PNG")
            st.download_button(
                "⬇ Grad-CAM (PNG)",
                data=buf.getvalue(),
                file_name=f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        except Exception:
            pass

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🍃</div>
        <div class="empty-title">Belum ada foto yang diunggah</div>
        <div class="empty-sub">
            Upload foto daun apel untuk memulai analisis kesehatan.<br>
            Format yang didukung: JPG, JPEG, PNG
        </div>
        <div style="display:flex;justify-content:center;gap:0.6rem;flex-wrap:wrap;">
            <span class="pill pill-green">EfficientNetB3</span>
            <span class="pill pill-mint">Grad-CAM XAI</span>
            <span class="pill pill-gray">4 Kelas Penyakit</span>
        </div>
    </div>""", unsafe_allow_html=True)