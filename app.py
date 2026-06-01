import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. DATABASE SAMPAH LENGKAP (10 KELAS - URUTAN ALFABETIS PYTORCH)

CLASS_NAMES = [
    'battery', 
    'biological', 
    'cardboard', 
    'clothes', 
    'glass', 
    'metal', 
    'paper', 
    'plastic', 
    'shoes', 
    'trash'
]

DATABASE_SAMPAH = {
    "biological": {
        "nama_id": "Sampah Biologis / Organik", 
        "golongan": "ORGANIK", 
        "saran": "Masukkan ke dalam wadah kompos untuk dijadikan pupuk organik tanaman. Pastikan tidak tercampur sampah anorganik agar proses pembusukan alami berjalan optimal."
    },
    "paper": {
        "nama_id": "Kertas", 
        "golongan": "ANORGANIK", 
        "saran": "Kumpulkan dalam kondisi kering dan tidak terkena noda minyak. Lipat rapi atau ikat bersama, lalu salurkan ke bank sampah atau pengepul untuk didaur ulang menjadi bubur kertas."
    },
    "cardboard": {
        "nama_id": "Kardus", 
        "golongan": "ANORGANIK", 
        "saran": "Pipihkan kardus untuk menghemat ruang penyimpanan. Pastikan bersih dari sisa lakban berlebih atau sisa makanan. Sangat bernilai tinggi di pusat daur ulang."
    },
    "plastic": {
        "nama_id": "Plastik", 
        "golongan": "ANORGANIK", 
        "saran": "Bilas sisa makanan/minuman yang menempel dan keringkan. Kelompokkan berdasarkan jenis botol atau kantong kresek sebelum diserahkan ke fasilitas daur ulang."
    },
    "metal": {
        "nama_id": "Logam / Kaleng", 
        "golongan": "ANORGANIK", 
        "saran": "Cuci bersih kaleng dari sisa cairan. Tekan atau pipihkan kaleng jika memungkinkan untuk efisiensi ruang. Logam dapat dilebur kembali tanpa menurunkan kualitasnya."
    },
    "glass": {
        "nama_id": "Kaca", 
        "golongan": "ANORGANIK", 
        "saran": "Pisahkan dengan hati-hati agar tidak percah dan melukai petugas kebersihan. Jika ada bagian yang retak/pecah, bungkus dengan koran tebal sebelum dibuang."
    },
    "clothes": {
        "nama_id": "Pakaian / Tekstil", 
        "golongan": "ANORGANIK", 
        "saran": "Jika masih layak pakai, pertimbangkan untuk didonasikan atau dijual sebagai pakaian bekas (thrifting). Jika rusak, potong menjadi kain lap atau bahan kerajinan tangan."
    },
    "shoes": {
        "nama_id": "Sepatu / Alas Kaki", 
        "golongan": "ANORGANIK", 
        "saran": "Jika masih bagus, bisa didonasikan atau diperbaiki di sol sepatu. Jika sudah rusak parah, pisahkan bagian karet solnya jika ingin dialihfungsikan oleh pengrajin karet/ban."
    },
    "battery": {
        "nama_id": "Baterai", 
        "golongan": "LIMBAH B3", 
        "saran": "SANGAT BERBAHAYA! Jangan dibakar, dihancurkan, atau dibuang bersama sampah rumah tangga biasa. Kumpulkan secara terpisah dan bawa ke dropbox limbah elektronik (e-waste) terdekat."
    },
    "trash": {
        "nama_id": "Sampah Residu / Umum", 
        "golongan": "RESIDU", 
        "saran": "Kategori sampah ini (seperti tisu bekas, pampers, puntung rokok, atau kemasan saset berlapis) sangat sulit didaur ulang secara mandiri. Buang ke tempat sampah utama untuk dialokasikan langsung ke TPA."
    }
}

# Load model secara efisien agar web tidak lambat (Cached)
@st.cache_resource
def load_model():
    # Inisialisasi arsitektur EfficientNet-B0 kosong
    model = models.efficientnet_b0(pretrained=False)
    # Ubah output layer sesuai dengan 10 kelas dataset
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    
    # Load bobot yang sudah di-commit sebelumnya
    model.load_state_dict(torch.load('efficientnet_b0_waste.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Transformasi gambar input agar sesuai standar training model (Resize 224x224 & Normalisasi ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. LOGIKAL FLOW INTERFACE 
st.set_page_config(page_title="Sistem Deteksi Sampah AI", layout="centered")

# Menggunakan Session State untuk mengatur transisi halaman
if 'page' not in st.session_state:
    st.session_state.page = 'upload_page'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# PAGE 1: UPLOAD GAMBAR 
if st.session_state.page == 'upload_page':
    st.title("♻️ Sistem Klasifikasi Sampah Cerdas")
    st.write("Silakan unggah foto sampah Anda untuk mendeteksi jenis golongan dan saran pengolahannya.")
    
    # Input File & Kamera sekaligus
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Atau ambil foto langsung")
    
    target_file = uploaded_file if uploaded_file is not None else camera_file
    
    if target_file is not None:
        st.session_state.uploaded_image = Image.open(target_file).convert('RGB')
        if st.button("Mulai Deteksi 🚀", use_container_width=True):
            st.session_state.page = 'result_page'
            st.rerun()

#  PAGE 2: HASIL DETEKSI 
elif st.session_state.page == 'result_page':
    st.title("📊 Hasil Analisis AI")
    
    if st.session_state.uploaded_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(st.session_state.uploaded_image, caption="Gambar yang Diunggah", use_container_width=True)
        
        with col2:
            with st.spinner("Model AI sedang menganalisis..."):
                # Proses prediksi gambar
                img_t = transform(st.session_state.uploaded_image).unsqueeze(0)
                model = load_model()
                
                with torch.no_grad():
                    outputs = model(img_t)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, index = torch.max(probabilities, 0)
                    
                predicted_class = CLASS_NAMES[index.item()]
                info_sampah = DATABASE_SAMPAH[predicted_class]
                
                # Tampilkan Informasi Hasil
                st.metric(label="Jenis Sampah Terdeteksi", value=info_sampah['nama_id'])
                
                # Warna dinamis berdasarkan golongan sampah
                golongan = info_sampah['golongan']
                if golongan == "ORGANIK":
                    st.success(f"Golongan: **{golongan}**")
                elif golongan == "ANORGANIK":
                    st.info(f"Golongan: **{golongan}**")
                elif golongan == "LIMBAH B3":
                    st.error(f"Golongan: **{golongan}**")
                else:
                    st.warning(f"Golongan: **{golongan}**")
                    
                st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence.item()*100:.2f}%")
        
        # Section Saran Pengolahan Sampah
        st.subheader("💡 Rekomendasi Cara Pengolahan:")
        st.info(info_sampah['saran'])
        
        # Tombol kembali (Reset)
        if st.button("🔄 Deteksi Gambar Lain", use_container_width=True):
            st.session_state.page = 'upload_page'
            st.session_state.uploaded_image = None
            st.rerun()
