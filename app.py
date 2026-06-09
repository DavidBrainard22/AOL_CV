import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── 1. DATA ───────────────────────────────────────────────────────────────────

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
        "icon": "🧴",
        "saran": "Bilas sisa makanan/minuman yang menempel dan keringkan. Kelompokkan berdasarkan jenis botol atau kantong kresek sebelum diserahkan ke fasilitas daur ulang."
    },
    "metal": {
        "nama_id": "Logam / Kaleng",
        "golongan": "ANORGANIK",
        "icon": "🥫",
        "saran": "Cuci bersih kaleng dari sisa cairan. Tekan atau pipihkan kaleng jika memungkinkan untuk efisiensi ruang. Logam dapat dilebur kembali tanpa menurunkan kualitasnya."
    },
    "glass": {
        "nama_id": "Kaca",
        "golongan": "ANORGANIK",
        "icon": "🫙",
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
        "icon": "🔋",
        "saran": "SANGAT BERBAHAYA! Jangan dibakar, dihancurkan, atau dibuang bersama sampah rumah tangga biasa. Kumpulkan secara terpisah dan bawa ke dropbox limbah elektronik (e-waste) terdekat."
    },
    "trash": {
        "nama_id": "Sampah Residu / Umum",
        "golongan": "RESIDU",
        "icon": "🗑️",
        "saran": "Kategori sampah ini (seperti tisu bekas, pampers, puntung rokok, atau kemasan saset berlapis) sangat sulit didaur ulang secara mandiri. Buang ke tempat sampah utama untuk dialokasikan langsung ke TPA."
    }
}

# ── 2. MODEL ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
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

# ── 3. PAGE CONFIG ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EcoScan — Klasifikasi Sampah AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── 4. GLOBAL STYLES ──────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(160deg, #E8F5E9 0%, #F0FDF4 60%, #ECFDF5 100%);
        min-height: 100vh;
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 3rem !important;
        max-width: 760px !important;
    }

    /* ── Header banner ── */
    .app-header {
        background: linear-gradient(135deg, #052e16 0%, #065f46 60%, #047857 100%);
        color: white;
        padding: 2.25rem 2rem 2rem;
        text-align: center;
        /* extend to screen edges */
        margin: -5rem -5rem 2rem -5rem;
        position: relative;
        overflow: hidden;
    }

    /* Decorative circles in header */
    .app-header::before {
        content: '';
        position: absolute;
        top: -50px; right: -50px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
        pointer-events: none;
    }
    .app-header::after {
        content: '';
        position: absolute;
        bottom: -70px; left: -30px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: rgba(255,255,255,0.04);
        pointer-events: none;
    }

    .header-emoji {
        font-size: 2.8rem;
        display: block;
        margin-bottom: 0.4rem;
        position: relative;
        z-index: 1;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .header-sub {
        font-size: 0.88rem;
        opacity: 0.75;
        margin: 0.35rem 0 0;
        position: relative;
        z-index: 1;
    }

    /* ── White info cards ── */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.07);
        margin-bottom: 0.85rem;
    }
    .card-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #059669;
        margin-bottom: 0.45rem;
    }
    .card-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #052e16;
        line-height: 1.35;
    }

    /* ── Golongan badge ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.77rem;
        letter-spacing: 0.6px;
        margin-top: 0.6rem;
    }
    .badge-organik   { background: #D1FAE5; color: #065f46; }
    .badge-anorganik { background: #DBEAFE; color: #1e40af; }
    .badge-b3        { background: #FEE2E2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-residu    { background: #FEF3C7; color: #92400e; }

    /* ── Confidence bar ── */
    .conf-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #059669;
        margin-bottom: 0.4rem;
    }
    .conf-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #052e16;
        line-height: 1.2;
        margin-bottom: 0.55rem;
    }
    .conf-bar-bg {
        background: #D1FAE5;
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, #059669, #34d399);
    }

    /* ── Recommendation card ── */
    .rec-card {
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        border-left: 4px solid #059669;
        margin-top: 0.25rem;
    }
    .rec-title {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #059669;
        margin-bottom: 0.65rem;
    }
    .rec-text {
        color: #1f2937;
        font-size: 0.93rem;
        line-height: 1.75;
        margin: 0;
    }

    /* ── OR divider ── */
    .or-divider {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin: 1rem 0;
        color: #9ca3af;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .or-divider::before, .or-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #e5e7eb;
    }

    /* ── Section title ── */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #052e16;
        margin: 0 0 0.4rem;
    }
    .section-sub {
        color: #6b7280;
        font-size: 0.88rem;
        margin: 0 0 1.25rem;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #6ee7b7;
        border-radius: 14px;
        padding: 0.5rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #059669;
    }

    /* ── Uploaded image rounded ── */
    [data-testid="stImage"] > img {
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }

    /* ── CTA Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #059669, #047857) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.97rem !important;
        letter-spacing: 0.15px !important;
        box-shadow: 0 4px 14px rgba(5,150,105,0.35) !important;
        transition: box-shadow 0.15s ease, transform 0.15s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(5,150,105,0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Spinner color ── */
    .stSpinner > div { border-top-color: #059669 !important; }
</style>
""", unsafe_allow_html=True)

# ── 5. SESSION STATE ──────────────────────────────────────────────────────────

if 'page' not in st.session_state:
    st.session_state.page = 'upload_page'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# ── 6. PERSISTENT HEADER ─────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <span class="header-emoji">♻️</span>
    <p class="header-title">EcoScan</p>
    <p class="header-sub">Deteksi & kelola sampah dengan kecerdasan buatan</p>
</div>
""", unsafe_allow_html=True)

# ── PAGE 1: UPLOAD ────────────────────────────────────────────────────────────

if st.session_state.page == 'upload_page':

    st.markdown('<p class="section-title">📤 Unggah Foto Sampah</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Pilih atau foto langsung sampah yang ingin diidentifikasi jenis dan cara pengelolaannya.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Pilih dari galeri", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="or-divider">atau ambil foto langsung</div>', unsafe_allow_html=True)

    camera_file = st.camera_input("Kamera", label_visibility="collapsed")

    target_file = uploaded_file if uploaded_file is not None else camera_file

    if target_file is not None:
        st.session_state.uploaded_image = Image.open(target_file).convert('RGB')
        st.image(
            st.session_state.uploaded_image,
            caption="Preview gambar",
            use_container_width=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Mulai Analisis", use_container_width=True):
            st.session_state.page = 'result_page'
            st.rerun()

# ── PAGE 2: RESULT ────────────────────────────────────────────────────────────

elif st.session_state.page == 'result_page':

    if st.session_state.uploaded_image is not None:

        col1, col2 = st.columns([1, 1], gap="medium")

        # --- run inference BEFORE entering columns so info is available below ---
        img_t = transform(st.session_state.uploaded_image).unsqueeze(0)
        model = load_model()

        with st.spinner("Model AI sedang menganalisis..."):
            with torch.no_grad():
                outputs = model(img_t)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, index = torch.max(probabilities, 0)

        predicted_class  = CLASS_NAMES[index.item()]
        info             = DATABASE_SAMPAH[predicted_class]
        confidence_pct   = confidence.item() * 100

        # Badge class
        badge_map = {
            "ORGANIK":   "badge-organik",
            "ANORGANIK": "badge-anorganik",
            "LIMBAH B3": "badge-b3",
            "RESIDU":    "badge-residu",
        }
        badge_cls = badge_map.get(info['golongan'], "badge-residu")

        with col1:
            st.image(
                st.session_state.uploaded_image,
                caption="Gambar yang dianalisis",
                use_container_width=True
            )

        with col2:
            # Detected waste card
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Jenis Terdeteksi</div>
                <div class="card-value">{info['icon']} {info['nama_id']}</div>
                <span class="badge {badge_cls}">● {info['golongan']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Confidence card with custom bar
            st.markdown(f"""
            <div class="card">
                <div class="conf-label">Tingkat Keyakinan</div>
                <div class="conf-val">{confidence_pct:.1f}%</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{confidence_pct:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Recommendation section
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">💡 Rekomendasi Pengelolaan</div>
            <p class="rec-text">{info['saran']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔄 Analisis Sampah Lain", use_container_width=True):
            st.session_state.page = 'upload_page'
            st.session_state.uploaded_image = None
            st.rerun()
