import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Disease Screener",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.header-bar {
    background: linear-gradient(90deg, #161b22 0%, #1c2a3a 100%);
    border-left: 4px solid #58a6ff;
    padding: 1.2rem 1.8rem;
    border-radius: 6px;
    margin-bottom: 1.8rem;
}
.header-bar h1 { color: #58a6ff; margin: 0; font-size: 1.6rem; letter-spacing: -0.5px; }
.header-bar p  { color: #8b949e; margin: 0.3rem 0 0; font-size: 0.85rem; }

.section-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 1rem;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
}

.risk-high {
    background: linear-gradient(135deg, #3d0c0c 0%, #5a1010 100%);
    border: 2px solid #f85149;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #0d2a1a 0%, #0f3622 100%);
    border: 2px solid #3fb950;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    text-align: center;
}
.risk-label-high { color: #f85149; font-size: 1.6rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.risk-label-low  { color: #3fb950; font-size: 1.6rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.confidence-text { color: #8b949e; font-size: 0.9rem; margin-top: 0.4rem; }
.confidence-pct  { color: #e6edf3; font-size: 1.2rem; font-weight: 600; }

.nurse-note {
    background: #1c2a3a;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    color: #c9d1d9;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-top: 1rem;
}
.hint { color: #6e7681; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; }

div[data-testid="stNumberInput"] label { font-size: 0.88rem; color: #c9d1d9; }
div[data-testid="stSelectbox"]   label { font-size: 0.88rem; color: #c9d1d9; }

div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 1px;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Load model artefacts ──────────────────────────────────────
@st.cache_resource
def load_artefacts():
    return joblib.load("dashboard_model.pkl")

artefacts     = load_artefacts()
model         = artefacts["model"]
scaler        = artefacts["scaler"]
feature_cols  = artefacts["feature_cols"]
CATEGORICAL   = artefacts["categorical"]
CONTINUOUS    = artefacts["continuous"]
test_patient  = artefacts["test_patient"]

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>🫀 CardioAI — Heart Disease Screening Dashboard</h1>
  <p>Decision-support tool for community cardiology clinics · Logistic Regression model · UCI Cleveland dataset</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: form (left) | results (right) ────────────────────
col_form, col_results = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Input — 13 Clinical Features</div>', unsafe_allow_html=True)
    st.markdown('<p class="hint">Pre-populated with a real test patient (true label: No Disease)</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age (years)", min_value=20, max_value=80,
            value=int(test_patient["age"]),
            help="Valid range: 20–80"
        )
        trestbps = st.number_input(
            "Resting Blood Pressure (mmHg)", min_value=80, max_value=220,
            value=int(test_patient["trestbps"]),
            help="Valid range: 80–220"
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)", min_value=100, max_value=600,
            value=int(test_patient["chol"]),
            help="Valid range: 100–600"
        )
        thalach = st.number_input(
            "Max Heart Rate Achieved", min_value=70, max_value=210,
            value=int(test_patient["thalach"]),
            help="Valid range: 70–210"
        )
        oldpeak = st.number_input(
            "ST Depression (oldpeak)", min_value=0.0, max_value=7.0,
            value=float(test_patient["oldpeak"]), step=0.1,
            help="Valid range: 0.0–7.0"
        )
        ca = st.number_input(
            "Major Vessels (fluoroscopy)", min_value=0, max_value=3,
            value=int(test_patient["ca"]),
            help="Number of major vessels coloured: 0–3"
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1], index=int(test_patient["fbs"]),
            format_func=lambda x: "Yes (1)" if x else "No (0)"
        )

    with c2:
        sex = st.selectbox(
            "Sex",
            options=[0, 1], index=int(test_patient["sex"]),
            format_func=lambda x: "Male (1)" if x else "Female (0)"
        )
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3], index=int(test_patient["cp"]),
            format_func=lambda x: {
                0: "0 — Typical Angina",
                1: "1 — Atypical Angina",
                2: "2 — Non-Anginal",
                3: "3 — Asymptomatic"
            }[x]
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2], index=int(test_patient["restecg"]),
            format_func=lambda x: {
                0: "0 — Normal",
                1: "1 — ST-T Abnormality",
                2: "2 — LV Hypertrophy"
            }[x]
        )
        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1], index=int(test_patient["exang"]),
            format_func=lambda x: "Yes (1)" if x else "No (0)"
        )
        slope = st.selectbox(
            "Slope of Peak ST Segment",
            options=[0, 1, 2], index=int(test_patient["slope"]),
            format_func=lambda x: {
                0: "0 — Upsloping",
                1: "1 — Flat",
                2: "2 — Downsloping"
            }[x]
        )
        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3], index=[1,2,3].index(int(test_patient["thal"])),
            format_func=lambda x: {
                1: "1 — Normal",
                2: "2 — Fixed Defect",
                3: "3 — Reversible Defect"
            }[x]
        )

    st.markdown("</div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  RUN PREDICTION")

# ── Prediction logic ──────────────────────────────────────────
def build_input_df(age, sex, cp, trestbps, chol, fbs,
                   restecg, thalach, exang, oldpeak, slope, ca, thal):
    raw = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    enc = pd.get_dummies(raw, columns=CATEGORICAL, drop_first=False)
    enc = enc.reindex(columns=feature_cols, fill_value=0)
    enc[CONTINUOUS] = scaler.transform(enc[CONTINUOUS])
    return enc

def feature_importance_chart(model, feature_cols, top_n=3):
    coefs = np.abs(model.coef_[0])
    series = pd.Series(coefs, index=feature_cols).sort_values(ascending=False)
    top = series.head(top_n)

    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    colors = ["#f85149", "#ff8c42", "#ffd166"]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], height=0.55)
    ax.set_xlabel("Absolute Coefficient", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color("#8b949e")
    plt.tight_layout(pad=0.5)
    return fig

def nurse_note(pred, proba, top_features):
    f1, f2, f3 = top_features[0], top_features[1], top_features[2]
    readable = {
        "thalach": "maximum heart rate", "oldpeak": "ST depression level",
        "ca": "number of blocked vessels", "age": "age",
        "chol": "cholesterol level", "trestbps": "resting blood pressure",
        "exang": "exercise-induced angina",
    }
    def r(f): return readable.get(f.split("_")[0], f.replace("_", " "))

    if pred == 1:
        return (
            f"This patient's {r(f1)}, {r(f2)}, and {r(f3)} are the strongest indicators "
            f"of elevated cardiac risk (model confidence: {proba*100:.1f}%). "
            f"Recommend referral for a full cardiac work-up including stress testing and echocardiography. "
            f"Do not discharge without cardiology review."
        )
    else:
        return (
            f"Based on {r(f1)}, {r(f2)}, and {r(f3)}, this patient's profile is consistent "
            f"with low cardiac risk (model confidence: {(1-proba)*100:.1f}%). "
            f"Routine follow-up is advised. If symptoms worsen, reassess with a repeat ECG and lipid panel."
        )

# ── Results panel ─────────────────────────────────────────────
with col_results:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        X_input = build_input_df(age, sex, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak, slope, ca, thal)
        pred    = int(model.predict(X_input)[0])
        proba   = float(model.predict_proba(X_input)[0][1])

        # Risk badge
        if pred == 1:
            st.markdown(f"""
            <div class="risk-high">
              <div class="risk-label-high">⚠ DISEASE PRESENT</div>
              <div class="confidence-text">Model confidence</div>
              <div class="confidence-pct">{proba*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
              <div class="risk-label-low">✓ NO DISEASE DETECTED</div>
              <div class="confidence-text">Model confidence</div>
              <div class="confidence-pct">{(1-proba)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Top 3 feature importances
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 3 Driving Features</div>', unsafe_allow_html=True)

        coefs      = np.abs(model.coef_[0])
        top3_feat  = pd.Series(coefs, index=feature_cols).sort_values(ascending=False).head(3)
        fig        = feature_importance_chart(model, feature_cols, top_n=3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Nurse note
        note = nurse_note(pred, proba, list(top3_feat.index))
        st.markdown(f'<div class="nurse-note">📋 <strong>Clinical note:</strong> {note}</div>',
                    unsafe_allow_html=True)

        # Raw probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Probability</div>', unsafe_allow_html=True)
        st.progress(proba)
        st.markdown(f'<p class="hint" style="text-align:center">P(disease) = {proba:.3f}</p>',
                    unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #6e7681;">
            <div style="font-size:3rem; margin-bottom:1rem;">🫀</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; letter-spacing:1px;">
                AWAITING INPUT
            </div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">
                Fill in the patient form and click RUN PREDICTION
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2rem; color:#6e7681; font-size:0.75rem;
            font-family:'IBM Plex Mono',monospace; border-top:1px solid #21262d; padding-top:1rem;">
    CardioAI Labs · Decision Support Only · Not a substitute for clinical judgement
</div>
""", unsafe_allow_html=True)
