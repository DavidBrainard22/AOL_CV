import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ============================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ============================================
st.set_page_config(
    page_title="WasteWise AI - Smart Waste Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    """Apply custom CSS for a professional, clean look."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 1rem 1rem;
    }
    
    /* Card styling */
    .card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eef2f6;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #10b981;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 40px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 16px;
        border: 2px dashed #cbd5e1;
        background-color: #f8fafc;
        padding: 1rem;
    }
    
    /* Camera input styling */
    .stCameraInput > div {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Progress bar color */
    .stProgress > div > div {
        background-color: #10b981;
        border-radius: 20px;
    }
    
    /* Info box custom */
    .custom-info {
        background-color: #eff6ff;
        border-radius: 16px;
        padding: 1rem;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ============================================
# 2. CONSTANTS & DATABASE
# ============================================
CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

DATABASE_SAMPAH = {
    "biological": {
        "nama_id": "Sampah Biologis / Organik",
        "golongan": "ORGANIK",
        "icon": "🌿",
        "saran": "Masukkan ke dalam wadah kompos untuk dijadikan pupuk organik tanaman. Pastikan tidak tercampur sampah anorganik agar proses pembusukan alami berjalan optimal."
    },
    "paper": {
        "nama_id": "Kertas",
        "golongan": "ANORGANIK",
        "icon": "📄",
        "saran": "Kumpulkan dalam kondisi kering dan tidak terkena noda minyak. Lipat rapi atau ikat bersama, lalu salurkan ke bank sampah atau pengepul untuk didaur ulang menjadi bubur kertas."
    },
    "cardboard": {
        "nama_id": "Kardus",
        "golongan": "ANORGANIK",
        "icon": "📦",
        "saran": "Pipihkan kardus untuk menghemat ruang penyimpanan. Pastikan bersih dari sisa lakban berlebih atau sisa makanan. Sangat bernilai tinggi di pusat daur ulang."
    },
    "plastic": {
        "nama_id": "Plastik",
        "golongan": "ANORGANIK",
        "icon": "🥤",
        "saran": "Bilas sisa makanan/minuman yang menempel dan keringkan. Kelompokkan berdasarkan jenis botol atau kantong kresek sebelum diserahkan ke fasilitas daur ulang."
    },
    "metal": {
        "nama_id": "Logam / Kaleng",
        "golongan": "ANORGANIK",
        "icon": "🔩",
        "saran": "Cuci bersih kaleng dari sisa cairan. Tekan atau pipihkan kaleng jika memungkinkan untuk efisiensi ruang. Logam dapat dilebur kembali tanpa menurunkan kualitasnya."
    },
    "glass": {
        "nama_id": "Kaca",
        "golongan": "ANORGANIK",
        "icon": "🥃",
        "saran": "Pisahkan dengan hati-hati agar tidak pecah dan melukai petugas kebersihan. Jika ada bagian yang retak/pecah, bungkus dengan koran tebal sebelum dibuang."
    },
    "clothes": {
        "nama_id": "Pakaian / Tekstil",
        "golongan": "ANORGANIK",
        "icon": "👕",
        "saran": "Jika masih layak pakai, pertimbangkan untuk didonasikan atau dijual sebagai pakaian bekas (thrifting). Jika rusak, potong menjadi kain lap atau bahan kerajinan tangan."
    },
    "shoes": {
        "nama_id": "Sepatu / Alas Kaki",
        "golongan": "ANORGANIK",
        "icon": "👟",
        "saran": "Jika masih bagus, bisa didonasikan atau diperbaiki di sol sepatu. Jika sudah rusak parah, pisahkan bagian karet solnya jika ingin dialihfungsikan oleh pengrajin karet/ban."
    },
    "battery": {
        "nama_id": "Baterai",
        "golongan": "LIMBAH B3",
        "icon": "⚠️",
        "saran": "SANGAT BERBAHAYA! Jangan dibakar, dihancurkan, atau dibuang bersama sampah rumah tangga biasa. Kumpulkan secara terpisah dan bawa ke dropbox limbah elektronik (e-waste) terdekat."
    },
    "trash": {
        "nama_id": "Sampah Residu / Umum",
        "golongan": "RESIDU",
        "icon": "🗑️",
        "saran": "Kategori sampah ini (seperti tisu bekas, pampers, puntung rokok, atau kemasan saset berlapis) sangat sulit didaur ulang secara mandiri. Buang ke tempat sampah utama untuk dialokasikan langsung ke TPA."
    }
}

# ============================================
# 3. MODEL LOADING (CACHED)
# ============================================
@st.cache_resource
def load_model():
    """Load the trained EfficientNet-B0 model."""
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load('efficientnet_b0_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# 4. SESSION STATE INITIALIZATION
# ============================================
if 'page' not in st.session_state:
    st.session_state.page = 'upload_page'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# ============================================
# 5. SIDEBAR (ABOUT & INFO)
# ============================================
with st.sidebar:
    st.markdown("## 🌍 Tentang WasteWise AI")
    st.markdown("""
    Aplikasi ini menggunakan **Deep Learning (EfficientNet-B0)** untuk mengidentifikasi 
    10 jenis sampah dari gambar. Dilengkapi dengan rekomendasi pengolahan sesuai golongan.
    """)
    st.markdown("### 📋 Kategori Sampah")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("• Battery (B3)\n• Biological\n• Cardboard\n• Clothes\n• Glass")
    with col2:
        st.markdown("• Metal\n• Paper\n• Plastic\n• Shoes\n• Trash")
    st.markdown("---")
    st.markdown("**💡 Catatan:** Model memiliki akurasi tinggi namun tetap disarankan verifikasi manual untuk limbah B3.")

# ============================================
# 6. PAGE LOGIC
# ============================================
# PAGE 1: UPLOAD
if st.session_state.page == 'upload_page':
    st.markdown("<h1 style='text-align: center;'>♻️ WasteWise AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #475569;'>Klasifikasi cerdas & rekomendasi daur ulang</p>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📸 Unggah Foto Sampah")
        st.markdown("Pilih gambar dari perangkat atau ambil langsung menggunakan kamera.")
        
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("📁 Upload dari komputer", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        with col2:
            camera_file = st.camera_input("📷 Ambil foto", label_visibility="collapsed")
        
        target_file = uploaded_file if uploaded_file is not None else camera_file
        
        if target_file is not None:
            st.session_state.uploaded_image = Image.open(target_file).convert('RGB')
            st.image(st.session_state.uploaded_image, caption="Gambar siap diproses", use_container_width=True)
            
            if st.button("🔍 Analisis Sampah", type="primary", use_container_width=True):
                st.session_state.page = 'result_page'
                st.rerun()
        else:
            st.info("Silakan unggah atau ambil foto untuk memulai deteksi.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">Powered by EfficientNet-B0 | Proyek Computer Vision - Semester 4</div>', unsafe_allow_html=True)

# PAGE 2: RESULTS
elif st.session_state.page == 'result_page':
    st.markdown("<h1 style='text-align: center;'>📊 Hasil Analisis</h1>", unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is not None:
        # Process prediction
        with st.spinner("🧠 AI sedang memproses gambar..."):
            img_tensor = transform(st.session_state.uploaded_image).unsqueeze(0)
            model = load_model()
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                top_probs, top_indices = torch.topk(probs, 3)
                
                confidence = top_probs[0].item()
                predicted_class = CLASS_NAMES[top_indices[0].item()]
                info = DATABASE_SAMPAH[predicted_class]
        
        # Layout: image and results in two columns
        col_img, col_res = st.columns([1, 1.2], gap="medium")
        
        with col_img:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, caption="Gambar yang diunggah", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_res:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Main result
            st.markdown(f"### {info['icon']} {info['nama_id']}")
            st.markdown(f"**Kelas (EN):** `{predicted_class}`")
            
            # Color-coded category
            golongan = info['golongan']
            if golongan == "ORGANIK":
                st.success(f"**Golongan:** {golongan} ♻️ Kompos")
            elif golongan == "ANORGANIK":
                st.info(f"**Golongan:** {golongan} ♻️ Daur Ulang")
            elif golongan == "LIMBAH B3":
                st.error(f"**Golongan:** {golongan} ⚠️ Bahaya")
            else:
                st.warning(f"**Golongan:** {golongan} 🗑️")
            
            # Confidence with progress bar
            st.markdown("**Tingkat Keyakinan AI**")
            st.progress(confidence, text=f"{confidence*100:.1f}%")
            
            # Top 3 predictions expander
            with st.expander("📊 Detail probabilitas kelas lain"):
                for i in range(3):
                    cls_name = CLASS_NAMES[top_indices[i].item()]
                    prob_val = top_probs[i].item()
                    st.markdown(f"**{i+1}. {cls_name}**  –  {prob_val*100:.1f}%")
                    st.progress(prob_val)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendation card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💡 Rekomendasi Pengolahan")
        st.markdown(f"<div class='custom-info'>{info['saran']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Deteksi Gambar Lain", use_container_width=True):
                st.session_state.page = 'upload_page'
                st.session_state.uploaded_image = None
                st.rerun()
        with col_btn2:
            if st.button("📸 Kembali ke Unggah", use_container_width=True):
                st.session_state.page = 'upload_page'
                st.session_state.uploaded_image = None
                st.rerun()
    
    else:
        st.warning("Tidak ada gambar ditemukan. Kembali ke halaman unggah.")
        if st.button("Ke Halaman Unggah"):
            st.session_state.page = 'upload_page'
            st.rerun()
    
    st.markdown('<div class="footer">♻️ Kelola sampah dengan bijak untuk lingkungan yang lebih baik.</div>', unsafe_allow_html=True)
