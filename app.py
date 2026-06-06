import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
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

.hero {
    background: linear-gradient(135deg, #1f4e79 0%, #2e86c1 60%, #1abc9c 100%);
    border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.8rem;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .3rem; }
.hero p  { font-size: 1rem; opacity: .85; margin: 0; }

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

.badge-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: 1rem; }
.badge {
    background: rgba(255,255,255,.18);
    border-radius: 8px;
    padding: .4rem .9rem;
    font-size: .82rem;
    font-weight: 600;
}

hr { border: none; border-top: 1.5px solid #e2e8f0; margin: 1.5rem 0; }

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


# ── Train model (cache agar hanya sekali saat startup) ────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model... mohon tunggu sebentar (~30 detik)")
def get_model():
    """
    Train model dari data CSV yang ada di repo.
    Menggunakan @st.cache_resource sehingga hanya dijalankan SEKALI
    saat app pertama kali dibuka, lalu disimpan di memory.
    """
    DATA_PATH = "Food_Delivery_Times.csv"

    if not os.path.exists(DATA_PATH):
        st.error(f"❌ File `{DATA_PATH}` tidak ditemukan di repo GitHub kamu!")
        st.info("Pastikan file CSV sudah di-upload ke GitHub bersama app.py")
        st.stop()

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=['Order_ID'], inplace=True, errors='ignore')

    # 2. Feature engineering
    df['distance_x_prep']    = df['Distance_km'] * df['Preparation_Time_min']
    df['exp_distance_ratio'] = df['Courier_Experience_yrs'] / (df['Distance_km'] + 1)
    df['total_time_proxy']   = df['Distance_km'] + df['Preparation_Time_min']

    # 3. Define features
    TARGET       = 'Delivery_Time_min'
    num_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs',
                    'distance_x_prep', 'exp_distance_ratio', 'total_time_proxy']
    cat_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']

    X = df[num_features + cat_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 4. Preprocessor
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features),
    ], remainder='drop')

    # 5. Train & pilih best model
    model_dict = {
        'Linear Regression' : LinearRegression(),
        'Decision Tree'     : DecisionTreeRegressor(random_state=42),
        'Random Forest'     : RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        'Gradient Boosting' : GradientBoostingRegressor(n_estimators=150, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        model_dict['XGBoost'] = XGBRegressor(
            n_estimators=150, random_state=42, verbosity=0, n_jobs=-1
        )

    best_r2   = -999
    best_pipe = None

    for name, mdl in model_dict.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', mdl)])
        pipe.fit(X_train, y_train)
        r2 = r2_score(y_test, pipe.predict(X_test))
        if r2 > best_r2:
            best_r2   = r2
            best_pipe = pipe

    # 6. Hyperparameter tuning XGBoost (jika tersedia)
    if XGBOOST_AVAILABLE:
        param_dist = {
            'model__n_estimators'    : [100, 200, 300, 400, 500],
            'model__max_depth'       : [3, 4, 5, 6, 7],
            'model__learning_rate'   : [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
            'model__subsample'       : [0.7, 0.8, 0.85, 0.9, 1.0],
            'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'model__reg_alpha'       : [0, 0.01, 0.05, 0.1, 0.5],
            'model__reg_lambda'      : [0.5, 1.0, 1.5, 2.0, 3.0],
            'model__min_child_weight': [1, 3, 5, 7],
        }
        xgb_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)),
        ])
        rscv = RandomizedSearchCV(
            xgb_pipe, param_dist, n_iter=50, cv=5,
            scoring='r2', random_state=42, n_jobs=-1, refit=True,
        )
        rscv.fit(X_train, y_train)
        if rscv.best_score_ > best_r2:
            best_pipe = rscv.best_estimator_

    return best_pipe


# Load / train model
model = get_model()


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
    distance_x_prep    = distance * prep_time
    exp_distance_ratio = courier_exp / (distance + 1)
    total_time_proxy   = distance + prep_time

    input_df = pd.DataFrame([{
        "Distance_km"            : distance,
        "Preparation_Time_min"   : float(prep_time),
        "Courier_Experience_yrs" : courier_exp,
        "distance_x_prep"        : distance_x_prep,
        "exp_distance_ratio"     : exp_distance_ratio,
        "total_time_proxy"       : total_time_proxy,
        "Weather"                : weather,
        "Traffic_Level"          : traffic,
        "Time_of_Day"            : time_of_day,
        "Vehicle_Type"           : vehicle,
    }])

    prediction = float(model.predict(input_df)[0])
    prediction = max(1.0, prediction)

    if prediction <= 30:
        speed_label = "⚡ Sangat Cepat"
    elif prediction <= 50:
        speed_label = "✅ Normal"
    elif prediction <= 70:
        speed_label = "⚠️ Agak Lama"
    else:
        speed_label = "🔴 Lama"

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

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">📊 Detail Kalkulasi Fitur</div>', unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Distance × Prep", f"{distance_x_prep:.1f}")
    dc2.metric("Exp/Distance Ratio", f"{exp_distance_ratio:.3f}")
    dc3.metric("Total Time Proxy", f"{total_time_proxy:.1f}")
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
