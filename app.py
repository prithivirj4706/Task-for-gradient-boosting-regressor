import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Premium CSS Styling ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d9ff, #00ff88, #ff00aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(0, 217, 255, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(0, 255, 136, 0.8)); }
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .feature-input-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-input-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(0, 217, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #00d9ff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-result {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 1rem;
    }
    
    .prediction-edible {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 200, 100, 0.1));
        border: 2px solid #00ff88;
    }
    
    .prediction-poisonous {
        background: linear-gradient(135deg, rgba(255, 0, 100, 0.2), rgba(200, 0, 80, 0.1));
        border: 2px solid #ff0064;
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .prediction-edible .prediction-text {
        color: #00ff88;
    }
    
    .prediction-poisonous .prediction-text {
        color: #ff0064;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d9ff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        color: #0f0c29;
        font-weight: 600;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #00d9ff;
        box-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 217, 255, 0.1);
        border-left: 4px solid #00d9ff;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d9ff;
    }
    
    .stat-label {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #a0aec0;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("gradient_boosting_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

try:
    model, feature_columns = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🍄 Navigation")
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    <div class='info-box'>
        <strong>Gradient Boosting Classifier</strong><br>
        This model predicts whether a mushroom is edible or poisonous based on its physical characteristics.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Info")
    if model_loaded:
        st.markdown(f"**Features:** {len(feature_columns)}")
        st.markdown(f"**Model Type:** Gradient Boosting")
    
    st.markdown("---")
    st.markdown("### Quick Tips")
    st.markdown("""
    - Enter all feature values
    - Click Predict to classify
    - Values default to 0.0
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    <div class='warning-box'>
        This is for educational purposes only. Never use this to determine if a real mushroom is safe to eat!
    </div>
    """, unsafe_allow_html=True)

# ---------------- Main Content ----------------
# Hero Section
st.markdown('<h1 class="hero-title">🍄 Mushroom Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Mushroom Classification using Gradient Boosting</p>', unsafe_allow_html=True)

# Stats Row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class='stat-card'>
        <div class='stat-value'>🧠</div>
        <div class='stat-label'>Gradient Boosting</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if model_loaded:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{len(feature_columns)}</div>
            <div class='stat-label'>Input Features</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-value'>--</div>
            <div class='stat-label'>Input Features</div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='stat-card'>
        <div class='stat-value'>2</div>
        <div class='stat-label'>Classes</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- User Input Section ----------------
if model_loaded:
    st.markdown('<div class="section-header">📊 Enter Mushroom Features</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    input_data = {}
    
    # Create a grid of inputs
    num_cols = 3
    feature_list = list(feature_columns)
    
    for i in range(0, len(feature_list), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            if i + j < len(feature_list):
                feature = feature_list[i + j]
                with col:
                    # Clean up feature name for display
                    display_name = feature.replace('_', ' ').title()
                    input_data[feature] = st.number_input(
                        f"🔹 {display_name}",
                        value=0.0,
                        key=feature,
                        help=f"Enter value for {display_name}"
                    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---------------- Prediction Button ----------------
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button("🔮 Classify Mushroom", use_container_width=True)
    
    # ---------------- Prediction Result ----------------
    if predict_clicked:
        with st.spinner("Analyzing mushroom characteristics..."):
            prediction = model.predict(input_df)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Determine result styling
            if prediction == 0 or str(prediction).lower() in ['edible', 'e', '0']:
                result_class = "prediction-edible"
                icon = "✅"
                result_text = "EDIBLE"
                message = "This mushroom appears to be safe based on the provided features."
            else:
                result_class = "prediction-poisonous"
                icon = "☠️"
                result_text = "POISONOUS"
                message = "WARNING: This mushroom appears to be toxic! Do not consume."
            
            st.markdown(f"""
            <div class='prediction-result {result_class}'>
                <div class='prediction-icon'>{icon}</div>
                <div class='prediction-text'>{result_text}</div>
                <p style='color: #a0aec0; margin-top: 1rem;'>{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show prediction details
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

else:
    st.error("⚠️ Model files not found. Please ensure `gradient_boosting_model.pkl` and `feature_columns.pkl` are in the same directory.")

# ---------------- Footer ----------------
st.markdown("""
<div class='footer'>
    <p>🍄 Mushroom Classifier | Powered by Gradient Boosting</p>
    <p style='font-size: 0.8rem;'>Team 2 Presentation Project | © 2026</p>
</div>
""", unsafe_allow_html=True)
