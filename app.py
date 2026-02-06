import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
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
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe, #00f2fe, #43e97b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(79, 172, 254, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(0, 242, 254, 0.8)); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
        animation: fadeInUp 0.6s ease-out;
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
        border-color: rgba(79, 172, 254, 0.3);
        transform: translateY(-2px);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4facfe;
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
        animation: fadeInUp 0.5s ease-out;
    }
    
    .prediction-low {
        background: linear-gradient(135deg, rgba(67, 233, 123, 0.2), rgba(56, 249, 215, 0.1));
        border: 2px solid #43e97b;
    }
    
    .prediction-medium {
        background: linear-gradient(135deg, rgba(250, 219, 95, 0.2), rgba(255, 193, 7, 0.1));
        border: 2px solid #ffc107;
    }
    
    .prediction-high {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(238, 77, 45, 0.1));
        border: 2px solid #ff6b6b;
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .prediction-low .prediction-text {
        color: #43e97b;
    }
    
    .prediction-medium .prediction-text {
        color: #ffc107;
    }
    
    .prediction-high .prediction-text {
        color: #ff6b6b;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(26, 26, 46, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #4facfe !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: #1a1a2e;
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
        box-shadow: 0 5px 20px rgba(79, 172, 254, 0.4);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4facfe;
        box-shadow: 0 0 10px rgba(79, 172, 254, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(79, 172, 254, 0.1);
        border-left: 4px solid #4facfe;
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
    
    .health-tip {
        background: rgba(67, 233, 123, 0.1);
        border-left: 4px solid #43e97b;
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
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4facfe;
    }
    
    .stat-label {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Progress bar animation */
    .progress-container {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease-out;
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
    
    /* Animated background particles effect */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
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
    st.markdown("## 🩺 Navigation")
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    <div class='info-box'>
        <strong>Gradient Boosting Regressor</strong><br>
        This app predicts the <strong>diabetes progression value</strong> based on patient health metrics using an advanced Gradient Boosting model.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Info")
    if model_loaded:
        st.markdown(f"**Features:** {len(feature_columns)}")
        st.markdown(f"**Model Type:** Gradient Boosting Regressor")
        st.markdown(f"**Output:** Continuous Value")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Reference")
    st.markdown("""
    **Common Features:**
    - **Age**: Patient's age
    - **BMI**: Body Mass Index
    - **BP**: Blood Pressure
    - **S1-S6**: Blood serum measurements
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Health Tips")
    st.markdown("""
    <div class='health-tip'>
        <strong>Prevention is key!</strong><br>
        Regular exercise, balanced diet, and routine check-ups can help manage diabetes risk.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    <div class='warning-box'>
        This is for educational purposes only. Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)

# ---------------- Main Content ----------------
# Hero Section
st.markdown('<h1 class="hero-title">🩺 Diabetes Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Diabetes Progression Prediction using Gradient Boosting Regression</p>', unsafe_allow_html=True)

# Stats Row
col1, col2, col3, col4 = st.columns(4)
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
        <div class='stat-value'>📈</div>
        <div class='stat-label'>Regression Model</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='stat-card'>
        <div class='stat-value'>⚡</div>
        <div class='stat-label'>Instant Prediction</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- User Input Section ----------------
if model_loaded:
    st.markdown('<div class="section-header">🔢 Enter Patient Health Data</div>', unsafe_allow_html=True)
    
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
                        f"💊 {display_name}",
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
        predict_clicked = st.button("🔍 Predict Diabetes Progression", use_container_width=True)
    
    # ---------------- Prediction Result ----------------
    if predict_clicked:
        with st.spinner("Analyzing patient data..."):
            prediction = model.predict(input_df)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Determine result styling based on prediction value
            # Assuming diabetes progression values - adjust thresholds as needed
            if prediction < 100:
                result_class = "prediction-low"
                icon = "💚"
                result_text = "LOW RISK"
                color = "#43e97b"
                message = "The predicted diabetes progression value is relatively low. Continue maintaining a healthy lifestyle!"
            elif prediction < 200:
                result_class = "prediction-medium"
                icon = "💛"
                result_text = "MODERATE RISK"
                color = "#ffc107"
                message = "The predicted value indicates moderate progression. Consider consulting with a healthcare professional."
            else:
                result_class = "prediction-high"
                icon = "❤️"
                result_text = "ELEVATED RISK"
                color = "#ff6b6b"
                message = "The predicted value is elevated. We strongly recommend consulting with a healthcare professional."
            
            st.markdown(f"""
            <div class='prediction-result {result_class}'>
                <div class='prediction-icon'>{icon}</div>
                <div class='prediction-text'>{result_text}</div>
                <div class='prediction-value' style='color: {color};'>{prediction:.2f}</div>
                <p style='color: #a0aec0; margin-top: 1rem; font-size: 1.1rem;'>{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual progress indicator
            max_val = 400  # Adjust based on your model's typical output range
            progress_pct = min(prediction / max_val * 100, 100)
            st.markdown(f"""
            <div class='progress-container'>
                <div class='progress-bar' style='width: {progress_pct}%; background: linear-gradient(90deg, #43e97b, #ffc107, #ff6b6b);'></div>
            </div>
            <p style='text-align: center; color: #a0aec0; margin-top: 0.5rem; font-size: 0.9rem;'>
                Predicted Progression: {prediction:.2f} / ~{max_val}
            </p>
            """, unsafe_allow_html=True)
            
            # Show prediction details
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
            
            # Health recommendations
            with st.expander("💡 Health Recommendations"):
                st.markdown("""
                <div style='color: #e0e0e0;'>
                    <h4 style='color: #4facfe;'>General Diabetes Prevention Tips:</h4>
                    <ul>
                        <li>🥗 <strong>Maintain a Balanced Diet</strong> - Focus on whole grains, lean proteins, and vegetables</li>
                        <li>🏃 <strong>Regular Physical Activity</strong> - Aim for at least 150 minutes of moderate exercise per week</li>
                        <li>⚖️ <strong>Manage Weight</strong> - Maintain a healthy BMI</li>
                        <li>🩺 <strong>Regular Check-ups</strong> - Monitor blood sugar levels periodically</li>
                        <li>😴 <strong>Adequate Sleep</strong> - Get 7-9 hours of quality sleep</li>
                        <li>🚭 <strong>Avoid Smoking</strong> - Smoking increases diabetes risk</li>
                    </ul>
                    <p style='margin-top: 1rem; font-style: italic; color: #a0aec0;'>
                        Remember: This prediction is for educational purposes only. Always consult with healthcare professionals for medical advice.
                    </p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("⚠️ Model files not found. Please ensure `gradient_boosting_model.pkl` and `feature_columns.pkl` are in the same directory.")

# ---------------- Footer ----------------
st.markdown("""
<div class='footer'>
    <p>🩺 Diabetes Prediction App | Powered by Gradient Boosting Regression</p>
    <p style='font-size: 0.8rem;'>Team 2 Presentation Project | © 2026</p>
</div>
""", unsafe_allow_html=True)
