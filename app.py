import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle

st.set_page_config(
    page_title="FruitAI — Quality Classifier",
    page_icon="🍎",
    layout="wide"
)

# ── CUSTOM CSS — Nature/Agriculture theme ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Hide default streamlit menu */
#MainMenu, footer { visibility: hidden; }

/* Hero title */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a8e6cf, #56ab2f, #f9ca24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-sub {
    text-align: center;
    color: #7fb3d3;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.1em;
    margin-bottom: 2rem;
}

/* Card style */
.fruit-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Result badges */
.badge-ripe {
    background: linear-gradient(135deg, #56ab2f, #a8e063);
    color: white;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    display: block;
    margin: 1rem 0;
    font-family: 'Playfair Display', serif;
}
.badge-fresh {
    background: linear-gradient(135deg, #1e90ff, #00bfff);
    color: white;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    display: block;
    margin: 1rem 0;
    font-family: 'Playfair Display', serif;
}
.badge-rotten {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    color: white;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    display: block;
    margin: 1rem 0;
    font-family: 'Playfair Display', serif;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: #e8f4f8 !important;
}

/* Sliders */
[data-testid="stSlider"] label {
    color: #a8d8ea !important;
    font-weight: 500;
}

/* Section headers */
.section-title {
    color: #a8e6cf;
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    border-bottom: 1px solid rgba(168,230,207,0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Progress bar for probability */
.prob-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 12px;
    margin: 4px 0 12px;
}
.prob-bar-fill-green {
    background: linear-gradient(90deg, #56ab2f, #a8e063);
    height: 12px;
    border-radius: 8px;
    transition: width 0.5s ease;
}
.prob-bar-fill-blue {
    background: linear-gradient(90deg, #1e90ff, #00bfff);
    height: 12px;
    border-radius: 8px;
}
.prob-bar-fill-red {
    background: linear-gradient(90deg, #c0392b, #e74c3c);
    height: 12px;
    border-radius: 8px;
}
.prob-label {
    color: #cde8f5;
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('model_name.pkl', 'rb') as f:
        model_name = pickle.load(f)
    return model, le, features, model_name

model, le, feature_names, model_name = load_all()

# ── HERO HEADER ───────────────────────────────────────────
st.markdown('<div class="hero-title">🍎 FruitAI Quality Classifier</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AGRICULTURE INTELLIGENCE · POWERED BY MACHINE LEARNING</div>',
            unsafe_allow_html=True)

# ── MODEL STATS BAR ───────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🤖 Algorithm",   model_name)
c2.metric("🍓 Classes",     "Fresh · Ripe · Rotten")
c3.metric("📊 Training data", "500 fruits")
c4.metric("🎯 Output type", "Multi-class")
st.divider()


# ── TWO COLUMN LAYOUT ─────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">📐 Fruit Measurements</div>',
                unsafe_allow_html=True)

    size = st.slider(
        "📏 Size (cm) — how big is the fruit?",
        3.0, 10.0, 6.5, step=0.1)

    weight = st.slider(
        "⚖️ Weight (g) — how heavy?",
        50.0, 300.0, 150.0, step=1.0)

    colour_score = st.slider(
        "🎨 Colour Score — 1=dull/green, 10=vibrant/perfect",
        1.0, 10.0, 5.0, step=0.1)

    firmness = st.slider(
        "💪 Firmness — 1=very soft, 10=very firm",
        1.0, 10.0, 5.0, step=0.1)

    sugar_content = st.slider(
        "🍬 Sugar Content — 1=not sweet, 10=very sweet",
        1.0, 10.0, 5.0, step=0.1)

    blemish_score = st.slider(
        "🔍 Blemish Score — 0=perfect, 10=very damaged",
        0.0, 10.0, 2.0, step=0.1)

    predict_btn = st.button(
        "🔬 Analyse Fruit Quality",
        type="primary",
        use_container_width=True)
    
with right:
    st.markdown('<div class="section-title">📊 Analysis Results</div>',
                unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="fruit-card">
            <div style="text-align:center; padding:2rem;">
                <div style="font-size:4rem;">🍎</div>
                <div style="color:#7fb3d3; margin-top:1rem;">
                    Adjust the measurements on the left<br>
                    and click <strong style="color:#a8e6cf;">Analyse Fruit Quality</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="fruit-card">
            <div style="color:#a8e6cf; font-weight:600; margin-bottom:0.8rem;">
                🌱 How it works
            </div>
            <div style="color:#cde8f5; font-size:0.9rem; line-height:1.7;">
                This AI model was trained on <strong>500 synthetic fruit samples</strong>
                using a <strong>Decision Tree</strong> algorithm.<br><br>
                It classifies fruits into 3 quality levels:<br>
                🔵 <strong>Fresh</strong> — firm, unripe, needs more time<br>
                🟢 <strong>Ripe</strong> — perfect quality, eat now<br>
                🔴 <strong>Rotten</strong> — damaged, do not consume
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── PREDICT ───────────────────────────────────────
        input_df = pd.DataFrame([[
            size, weight, colour_score,
            firmness, sugar_content, blemish_score
        ]], columns=feature_names)

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        quality     = le.inverse_transform([prediction])[0]

        # ── RESULT BADGE ──────────────────────────────────
        if quality == 'Ripe':
            st.markdown(
                '<span class="badge-ripe">🟢 RIPE — Perfect Quality!</span>',
                unsafe_allow_html=True)
        elif quality == 'Fresh':
            st.markdown(
                '<span class="badge-fresh">🔵 FRESH — Needs More Time</span>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="badge-rotten">🔴 ROTTEN — Do Not Consume</span>',
                unsafe_allow_html=True)
            
        # ── PROBABILITY BARS ──────────────────────────────
        st.markdown('<div class="section-title">Confidence Breakdown</div>',
                    unsafe_allow_html=True)

        class_config = {
            'Fresh' : ('#1e90ff', 'prob-bar-fill-blue',  '🔵'),
            'Ripe'  : ('#56ab2f', 'prob-bar-fill-green', '🟢'),
            'Rotten': ('#c0392b', 'prob-bar-fill-red',   '🔴'),
        }

        for cls in le.classes_:
            idx  = list(le.classes_).index(cls)
            pct  = round(float(probability[idx]) * 100, 1)
            cfg  = class_config[cls]
            st.markdown(f"""
            <div class="prob-label">
                <span>{cfg[2]} {cls}</span>
                <span style="font-weight:600;">{pct}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="{cfg[1]}" style="width:{pct}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── SCORE BREAKDOWN ───────────────────────────────
        st.markdown('<div class="section-title">Your Fruit\'s Scores</div>',
                    unsafe_allow_html=True)

        scores = {
            "📏 Size"         : (size,          3.0,  10.0),
            "⚖️ Weight"       : (weight,        50.0, 300.0),
            "🎨 Colour"       : (colour_score,   1.0,  10.0),
            "💪 Firmness"     : (firmness,        1.0,  10.0),
            "🍬 Sugar"        : (sugar_content,   1.0,  10.0),
            "🔍 Blemish"      : (blemish_score,   0.0,  10.0),
        }

        for label, (val, mn, mx) in scores.items():
            pct = int((val - mn) / (mx - mn) * 100)
            # For blemish — lower is better so flip color
            if "Blemish" in label:
                color = "#e74c3c" if pct > 60 else (
                        "#f39c12" if pct > 30 else "#56ab2f")
            else:
                color = "#56ab2f" if pct > 60 else (
                        "#f39c12" if pct > 30 else "#e74c3c")
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        color:#cde8f5; font-size:0.88rem;
                        margin-bottom:2px;">
                <span>{label}</span>
                <span style="color:{color}; font-weight:600;">{val}</span>
            </div>
            <div class="prob-bar-bg">
                <div style="background:{color}; width:{pct}%;
                            height:8px; border-radius:8px;"></div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── SUGGESTIONS ───────────────────────────────────
        st.markdown('<div class="section-title">💡 Recommendation</div>',
                    unsafe_allow_html=True)

        if quality == 'Ripe':
            st.success("🌟 Perfect fruit! Best consumed within 1-2 days for optimal taste and nutrition.")
        elif quality == 'Fresh':
            st.info("⏳ This fruit needs 2-3 more days to ripen. Store at room temperature away from direct sunlight.")
        else:
            tips = []
            if firmness < 3:
                tips.append("💪 Very low firmness — fruit has gone soft and mushy.")
            if blemish_score > 7.5:
                tips.append("🔍 Very high blemish score — too much surface damage.")
            if colour_score < 2.5:
                tips.append("🎨 Poor colour score — signs of decay or disease.")
            for t in tips:
                st.warning(t)

        # ── FEATURE IMPORTANCE ────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-title" style="text-align:center;">📊 Which Properties Matter Most?</div>',
                    unsafe_allow_html=True)

        imp_df = pd.DataFrame({
            'Property'  : feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        st.bar_chart(imp_df.set_index('Property'),
                    color="#56ab2f")

        st.caption(f"Model: {model_name} · Trained on 500 fruit samples · Multi-class classification")