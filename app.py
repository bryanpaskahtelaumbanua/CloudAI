import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import plotly.express as px

# === 1. CONFIG UI ===
st.set_page_config(
    page_title="CloudWhisper AI",
    page_icon="☁️",
    layout="wide", # Menggunakan layout wide untuk memanfaatkan lebar layar
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tema cuaca yang menarik dan responsif
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #87CEEB 0%, #ADD8E6 50%, #B0E0E6 100%); /* Gradasi biru langit */
        font-family: 'Poppins', sans-serif;
    }
    .main {
        padding: 2rem 3rem; /* Padding lebih besar */
        background: rgba(255, 255, 255, 0.9); /* Latar belakang putih transparan */
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    h1 {
        color: #2F80ED; /* Biru cerah */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #333333;
        font-weight: 600;
    }
    .stFileUploader label {
        color: #2F80ED;
        font-size: 1.1em;
        font-weight: 600;
    }
    .stFileUploader > div > button {
        background-color: #2F80ED;
        color: white;
        border-radius: 10px;
        padding: 0.7em 1.5em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > button:hover {
        background-color: #56CCF2;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stImage {
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    .stMetric {
        background-color: #FFFFFF;
        padding: 15px 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #E0E0E0;
        margin-bottom: 15px;
    }
    .stMetric > div > div:first-child { /* Label */
        color: #555555;
        font-weight: 600;
        font-size: 0.9em;
    }
    .stMetric > div > div:nth-child(2) { /* Value */
        color: #2F80ED;
        font-weight: 700;
        font-size: 1.8em;
    }
    .stMetric > div > div:nth-child(3) { /* Delta */
        color: #666666;
    }
    .stProgress > div > div {
        background-color: #56CCF2; /* Warna progress bar */
    }
    .stSpinner > div {
        color: #2F80ED;
    }
    .stMarkdown a {
        color: #2F80ED;
        text-decoration: none;
    }
    .stMarkdown a:hover {
        text-decoration: underline;
    }
    .css-1d391kg { /* Ini biasanya sidebar, menyesuaikan warnanya */
        background: linear-gradient(180deg, #ADD8E6 0%, #B0E0E6 100%);
    }
    .sidebar .st-emotion-cache-1pxx5pb { /* Streamlit header di sidebar */
        color: #2F80ED;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# === 2. LOAD MODEL & UTILS ===
MODEL_PATH = "model_awan.pth" # Pastikan file model ada di direktori yang sama
CLASSES = [
    "Altostratus", "Altocumulus", "Cirrus", "Cirrostratus", "Cirrocumulus",
    "Cumulonimbus", "Cumulus", "Nimbostratus", "Stratocumulus", "Stratus", "Towering Cumulus"
]

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 11)
    # Gunakan weights_only=True untuk keamanan di versi terbaru torch jika diperlukan
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standar ImageNet
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor)
        probs = F.softmax(out, dim=1)[0]
    return probs

# === 3. MAIN APP LAYOUT (ONE PAGE) ===

# Header Aplikasi
st.title("☁️ CloudWhisper AI: Prediksi Awan Cerdas")
st.markdown("### Unggah gambar awan Anda dan biarkan AI kami mengidentifikasinya!")

# Kolom utama untuk input dan output
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("---")
    st.subheader("📸 Unggah Gambar Awan")
    st.write("Seret dan letakkan file gambar (JPG, PNG) di sini, atau klik tombol di bawah.")
    uploaded = st.file_uploader("Pilih File Gambar", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True, width=300)
    else:
        st.info("👆 Belum ada gambar yang diunggah. Coba unggah satu!")
        # Placeholder gambar awan jika belum ada yang diupload
        st.image("https://images.unsplash.com/photo-1594902120023-e578a54162e7?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8Y2xvdWRzfHx8fHx8MTY5OTgxMDcyMA&ixlib=rb-4.0.3&q=80&w=1080", 
                 caption="Contoh Gambar Awan", use_container_width=True, width=300)
        

with col2:
    st.markdown("---")
    st.subheader("💡 Hasil Analisis AI")
    if uploaded:
        model = load_model()
        
        with st.spinner("☁️ AI sedang menganalisis pola awan Anda..."):
            probs = predict(image, model)
            conf, pred_idx = torch.max(probs, 0)
            class_name = CLASSES[pred_idx]
            confidence_val = conf.item() * 100

        st.success("🎉 Prediksi Selesai!")
        
        # Metrik Utama
        st.write("") # Spacer
        st.metric("Prediksi Utama", class_name, help="Jenis awan dengan probabilitas tertinggi.")
        st.metric("Tingkat Keyakinan", f"{confidence_val:.1f}%", help="Persentase keyakinan AI terhadap prediksi.")

        # Progress bar untuk visualisasi cepat
        st.progress(conf.item())

        st.write("---")
        st.subheader("Detail Probabilitas Klasifikasi")
        
        # Plot Interaktif dengan Plotly
        df = pd.DataFrame({
            "Jenis Awan": CLASSES,
            "Probabilitas": probs.numpy()
        }).sort_values("Probabilitas", ascending=True)

        fig = px.bar(df, x="Probabilitas", y="Jenis Awan", orientation='h',
                     title="Distribusi Probabilitas untuk Setiap Jenis Awan",
                     color="Probabilitas",
                     color_continuous_scale="Blues_r", # Skala warna kebiruan terbalik
                     height=450)
        
        fig.update_layout(showlegend=False, 
                          margin=dict(l=0, r=0, t=50, b=0),
                          xaxis_title="Probabilitas (%)",
                          yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("☁️ Unggah gambar awan di kolom kiri untuk melihat hasil prediksi di sini.")
        st.write("")
        st.markdown("""
        **Bagaimana cara kerjanya?**
        CloudWhisper AI menggunakan model **ResNet-18** yang telah dilatih secara khusus untuk mengenali berbagai formasi awan. Setiap detail di gambar Anda dianalisis untuk menemukan pola yang cocok dengan jenis awan yang berbeda.
        """)

# Bagian "Tentang Kami" atau FAQ di bawah (Opsional, dalam expander)
st.markdown("---")
with st.expander("📚 Pelajari Lebih Lanjut Tentang Jenis Awan"):
    st.markdown("""
    Awan memainkan peran vital dalam siklus air dan cuaca global. Memahami jenis-jenis awan dapat memberikan petunjuk tentang kondisi atmosfer saat ini dan yang akan datang.

    Berikut adalah beberapa jenis awan yang dapat diidentifikasi oleh CloudWhisper AI:
    * **Altostratus:** Awan berlapis abu-abu atau kebiruan yang menutupi seluruh langit, biasanya menandakan cuaca cerah di masa depan.
    * **Altocumulus:** Massa awan putih ke abu-abuan, berlapis-lapis atau bergulir, dengan dasar yang lebih rendah dari awan tinggi.
    * **Cirrus:** Awan tinggi, tipis, berserat, dan berwarna putih yang sering disebut "ekor kuda" karena bentuknya. Menunjukkan cuaca cerah.
    * **Cirrostratus:** Awan tipis, keputihan, seperti selimut yang menutupi seluruh langit, sering menghasilkan "halo" di sekitar matahari atau bulan.
    * **Cirrocumulus:** Lapisan tipis awan putih yang terdiri dari gumpalan kecil, seperti sisik ikan. Jarang terlihat.
    * **Cumulonimbus:** Awan badai vertikal yang besar, gelap, dan menjulang tinggi, membawa hujan lebat, petir, dan guntur.
    * **Cumulus:** Awan putih, menggumpal, berbulu kapas, biasanya menandakan cuaca cerah.
    * **Nimbostratus:** Awan berlapis abu-abu gelap, tebal, dan tidak berbentuk yang menghasilkan hujan atau salju berkelanjutan.
    * **Stratocumulus:** Awan rendah, abu-abu atau keputihan, dengan area gelap yang terpisah. Menunjukkan cuaca kering atau hujan ringan.
    * **Stratus:** Awan datar, berlapis-lapis, seragam, dan abu-abu yang menutupi langit, sering membawa gerimis ringan atau kabut.
    * **Towering Cumulus:** Bentuk cumulus yang lebih tinggi dan vertikal, bisa berkembang menjadi cumulonimbus.
    """, unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Cloud_types.jpg/1200px-Cloud_types.jpg", caption="Berbagai Jenis Awan", use_container_width=True)


# FOOTER Aplikasi
st.markdown(
    """
    <div style='text-align: center; color: #666666; padding-top: 3rem;'>
        <small>Dibuat dengan ❤️ oleh Tim AI Enthusiasts | Powered by Streamlit & PyTorch</small><br>
        <small>Versi 1.0.0 | <a href="https://github.com/your-github-repo" target="_blank">Kode Sumber</a></small>
    </div>
    """, 
    unsafe_allow_html=True
)
