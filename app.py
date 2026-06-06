import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery ETA Predictor",
    page_icon="🛵",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Header hero */
.hero {
    background: linear-gradient(135deg, #1f4e79 0%, #2e86c1 60%, #1abc9c 100%);
    border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.8rem;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .3rem; }
.hero p  { font-size: 1rem; opacity: .85; margin: 0; }

/* Section cards */
.card {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: .8rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: .9rem;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #1f4e79, #1abc9c);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    color: white;
    margin-top: 1.2rem;
}
.result-box .eta-number {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
}
.result-box .eta-label {
    font-size: 1rem;
    opacity: .85;
    margin-top: .3rem;
}

/* Metric badges */
.badge-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: 1rem; }
.badge {
    background: rgba(255,255,255,.18);
    border-radius: 8px;
    padding: .4rem .9rem;
    font-size: .82rem;
    font-weight: 600;
}

/* Divider */
hr { border: none; border-top: 1.5px solid #e2e8f0; margin: 1.5rem 0; }

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: .78rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "best_model.joblib"
    if not os.path.exists(model_path):
        st.error("❌ File `best_model.joblib` tidak ditemukan! "
                 "Pastikan file ada di folder yang sama dengan app.py.")
        st.stop()
    return joblib.load(model_path)

model = load_model()


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛵 Food Delivery ETA Predictor</h1>
    <p>Prediksi estimasi waktu pengiriman makanan berbasis Machine Learning (XGBoost).<br>
       IS411 Data Modelling · Grup 06 · Universitas Multimedia Nusantara</p>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📍 Detail Pengiriman</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    distance = st.number_input(
        "Jarak Pengiriman (km)", min_value=0.5, max_value=50.0,
        value=10.0, step=0.5,
        help="Jarak dari restoran ke lokasi pelanggan"
    )
    prep_time = st.number_input(
        "Waktu Persiapan Makanan (menit)", min_value=1, max_value=60,
        value=15, step=1,
        help="Berapa menit makanan disiapkan oleh restoran"
    )
    courier_exp = st.number_input(
        "Pengalaman Kurir (tahun)", min_value=0.0, max_value=15.0,
        value=3.0, step=0.5,
        help="Lama pengalaman kurir bekerja"
    )

with col2:
    weather = st.selectbox(
        "Kondisi Cuaca",
        options=["Clear", "Cloudy", "Foggy", "Rainy", "Snowy", "Windy"],
        index=0
    )
    traffic = st.selectbox(
        "Tingkat Kemacetan",
        options=["Low", "Medium", "High"],
        index=1
    )
    time_of_day = st.selectbox(
        "Waktu Pengiriman",
        options=["Morning", "Afternoon", "Evening", "Night"],
        index=2
    )
    vehicle = st.selectbox(
        "Jenis Kendaraan Kurir",
        options=["Bike", "Motorcycle", "Scooter"],
        index=1
    )

st.markdown('</div>', unsafe_allow_html=True)


# ── Predict button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Prediksi Waktu Pengiriman", use_container_width=True, type="primary")

if predict_btn:
    # Feature engineering (harus sama persis dengan saat training)
    distance_x_prep    = distance * prep_time
    exp_distance_ratio = courier_exp / (distance + 1)
    total_time_proxy   = distance + prep_time

    input_df = pd.DataFrame([{
        "Distance_km"            : distance,
        "Preparation_Time_min"   : prep_time,
        "Courier_Experience_yrs" : courier_exp,
        "distance_x_prep"        : distance_x_prep,
        "exp_distance_ratio"     : exp_distance_ratio,
        "total_time_proxy"       : total_time_proxy,
        "Weather"                : weather,
        "Traffic_Level"          : traffic,
        "Time_of_Day"            : time_of_day,
        "Vehicle_Type"           : vehicle,
    }])

    prediction = model.predict(input_df)[0]
    prediction = max(1.0, prediction)  # tidak boleh negatif

    # Tentukan kategori kecepatan
    if prediction <= 30:
        speed_label = "⚡ Sangat Cepat"
        speed_color = "#1abc9c"
    elif prediction <= 50:
        speed_label = "✅ Normal"
        speed_color = "#2e86c1"
    elif prediction <= 70:
        speed_label = "⚠️ Agak Lama"
        speed_color = "#f39c12"
    else:
        speed_label = "🔴 Lama"
        speed_color = "#e74c3c"

    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:1rem; opacity:.85; margin-bottom:.5rem;">Estimasi Waktu Pengiriman</div>
        <div class="eta-number">{prediction:.0f}</div>
        <div class="eta-label">menit · {speed_label}</div>
        <div class="badge-row" style="justify-content:center; margin-top:1.2rem;">
            <div class="badge">📍 {distance} km</div>
            <div class="badge">🍳 {prep_time} mnt prep</div>
            <div class="badge">🌤️ {weather}</div>
            <div class="badge">🚦 Traffic: {traffic}</div>
            <div class="badge">🛵 {vehicle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown detail
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">📊 Detail Kalkulasi Fitur</div>', unsafe_allow_html=True)
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("Distance × Prep", f"{distance_x_prep:.1f}")
    detail_col2.metric("Exp/Distance Ratio", f"{exp_distance_ratio:.3f}")
    detail_col3.metric("Total Time Proxy", f"{total_time_proxy:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)


# ── Info section ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ Tentang Model & Fitur"):
    st.markdown("""
    **Model:** XGBoost Regressor dengan hyperparameter tuning (RandomizedSearchCV, 50 iterasi × 5-fold CV)

    **Fitur Input:**

    | Fitur | Tipe | Keterangan |
    |---|---|---|
    | Distance_km | Numerik | Jarak pengiriman |
    | Preparation_Time_min | Numerik | Waktu persiapan makanan |
    | Courier_Experience_yrs | Numerik | Pengalaman kurir |
    | Weather | Kategorikal | Kondisi cuaca |
    | Traffic_Level | Kategorikal | Tingkat kemacetan |
    | Time_of_Day | Kategorikal | Waktu pengiriman |
    | Vehicle_Type | Kategorikal | Jenis kendaraan |

    **Fitur Engineered (otomatis dihitung):**
    - `distance_x_prep` = Distance × Preparation Time
    - `exp_distance_ratio` = Experience / (Distance + 1)
    - `total_time_proxy` = Distance + Preparation Time

    **Preprocessing:** SimpleImputer → StandardScaler (numerik), OneHotEncoder (kategorikal)

    **Target:** `Delivery_Time_min` — waktu pengiriman dalam menit
    """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    IS411 Data Modelling (Lab) · Ujian Akhir Semester · Semester Genap 2024/2025<br>
    Grup 06 · Universitas Multimedia Nusantara
</div>
""", unsafe_allow_html=True)
