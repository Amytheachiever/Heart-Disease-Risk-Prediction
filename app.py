import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Set page config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .factor-box {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .factor-box:hover {
        transform: translateY(-2px);
    }
    .factor-1 {
        background-color: #fff5f5;
        border-left: 6px solid #ff4b4b;
    }
    .factor-2 {
        background-color: #fff0f6;
        border-left: 6px solid #ff85c0;
    }
    .factor-3 {
        background-color: #f0f7ff;
        border-left: 6px solid #1890ff;
    }
    .factor-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .factor-impact {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.8rem;
    }
    .factor-recommendation {
        font-size: 1rem;
        line-height: 1.5;
        color: #444;
    }
    .recommendations-header {
        color: #2c3e50;
        font-size: 1.4rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# Factor-specific recommendations
FACTOR_RECOMMENDATIONS = {
    'age': "Stay active, eat healthy, and get regular check-ups to mitigate age-related risks.",
    'sex': "Monitor gender-specific risks (e.g., hormonal changes in women post-menopause).",
    'cp': "Consult a doctor for proper diagnosis; manage stress & avoid triggers.",
    'trestbps': "Reduce salt, exercise, manage stress, and take prescribed meds if needed.",
    'chol': "Eat more fiber, healthy fats, and exercise; consider statins if advised.",
    'fbs': "Cut sugar/refined carbs, exercise, and monitor glucose levels.",
    'restecg': "Follow up with a cardiologist for abnormal readings.",
    'thalach': "Improve cardio fitness with regular aerobic exercise.",
    'exang': "Avoid overexertion; follow a supervised cardiac rehab plan.",
    'oldpeak': "Seek medical evaluation for ischemia; manage BP & cholesterol.",
    'slope': "Requires medical assessment; maintain heart-healthy habits.",
    'ca': "Lifestyle changes + possible surgical intervention if blocked.",
    'thal': "Follow a cardiologist's advice for blood disorder management."
}

# Title and description
st.title("💓 Heart Disease Risk Prediction App")
st.markdown("""
    This app helps you assess your risk of developing heart disease based on various health metrics. 
    Enter your information below to get a personalized risk assessment and recommendations.
""")

# Sidebar for additional information
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    This prediction model uses machine learning to assess heart disease risk based on:
    - Basic health metrics
    - Medical history
    - Lifestyle factors
    
    The model provides:
    - Risk level assessment
    - Probability score
    - Feature importance analysis
    - Personalized recommendations
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    with st.form("user_input"):
        st.subheader("📝 Health Information")
        
        # Age with validation
        age = st.slider("Age", 20, 90, 50)
        if age > 65:
            st.warning("⚠️ Advanced age is a risk factor for heart disease. Regular check-ups are recommended.")
        
        # Sex selection with explanation
        sex = st.radio("Sex", ["Male", "Female"])
        if sex == "Male":
            st.info("Men are generally at higher risk of heart disease at younger ages.")
        
        # Chest Pain Type with detailed explanation
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[x]
        )
        
        # Blood Pressure with validation
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
        if trestbps > 140:
            st.warning("⚠️ Your blood pressure is above normal range. Please consult a doctor.")
        
        # Cholesterol with validation
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        if chol > 240:
            st.warning("⚠️ Your cholesterol level is high. Consider lifestyle changes and consult a doctor.")
        
        # Other inputs with explanations
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        if fbs == "Yes":
            st.warning("⚠️ High fasting blood sugar may indicate diabetes risk.")
        
        restecg = st.selectbox(
            "Resting ECG Results",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )
        
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        
        slope = st.selectbox(
            "Slope of ST Segment",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )
        
        ca = st.slider("Number of Major Vessels Colored", 0, 4, 0)
        thal = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Normal",
                1: "Fixed Defect",
                2: "Reversible Defect",
                3: "Not Applicable"
            }[x]
        )

        submitted = st.form_submit_button("🔍 Predict Risk")

# Convert inputs into dataframe
if submitted:
    input_data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "Yes" else 0,
        'restecg': restecg,
        'thalach': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    input_df = pd.DataFrame([input_data])

    # Prediction and Probability
    prob = model.predict_proba(input_df)[0][1]
    prob_percent = round(prob * 100, 2)

    # Risk categorization
    if prob_percent <= 20:
        risk_level = "🟢 Very Low Risk"
        style = st.success
    elif prob_percent <= 40:
        risk_level = "🟢 Low Risk"
        style = st.success
    elif prob_percent <= 60:
        risk_level = "🟡 Medium Risk"
        style = st.warning
    elif prob_percent <= 80:
        risk_level = "🟠 High Risk"
        style = st.error
    else:
        risk_level = "🔴 Very High Risk"
        style = st.error

    # SHAP analysis
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    
    # Get top 3 most influential features
    feature_importance = pd.DataFrame({
        'feature': input_df.columns,
        'importance': np.abs(shap_values.values[0])
    }).sort_values('importance', ascending=False)
    
    top_3_factors = feature_importance.head(3)

    # Display results in a more organized way
    with col2:
        st.subheader("📊 Risk Assessment Results")
        style(f"**{risk_level}**\n\n🔢 Probability: `{prob_percent}%`")
        
        st.markdown('<div class="recommendations-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
        st.markdown("Based on your top 3 most influential risk factors:")
        
        for idx, (_, row) in enumerate(top_3_factors.iterrows(), 1):
            factor = row['feature']
            importance = row['importance']
            st.markdown(f"""
            <div class="factor-box factor-{idx}">
                <div class="factor-title">{factor.upper()}</div>
                <div class="factor-impact">Impact Score: {importance:.2f}</div>
                <div class="factor-recommendation">{FACTOR_RECOMMENDATIONS[factor]}</div>
            </div>
            """, unsafe_allow_html=True)

    # SHAP analysis with better explanations
    st.subheader("📊 Model Explanation")
    
    # Summary Plot with better formatting
    st.markdown("### 🔍 Global Feature Importance")
    st.markdown("This shows which factors most influence the model's predictions across all patients.")
    fig_summary, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig_summary)

    # Waterfall plot with better explanation
    st.markdown("### 💡 Your Personal Risk Factors")
    st.markdown("This shows how each of your specific health metrics contributes to your risk score.")
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.plots.waterfall(shap_values[0], max_display=13, show=False)
    plt.tight_layout()
    st.pyplot(fig)

    # Add disclaimer
    st.markdown("""
    ---
    *Disclaimer: This tool is for informational purposes only and should not be considered medical advice. 
    Always consult with healthcare professionals for medical decisions.*
    """)









