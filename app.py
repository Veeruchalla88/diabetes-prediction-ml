import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── PAGE CONFIG 
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ── CUSTOM CSS 
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #1B4F8A;
        color: white;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    .stButton>button:hover { background-color: #154070; }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .diabetic {
        background-color: #FADBD8;
        border: 2px solid #E74C3C;
        color: #C0392B;
    }
    .not-diabetic {
        background-color: #D5F5E3;
        border: 2px solid #27AE60;
        color: #1E8449;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1B4F8A;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL 
@st.cache_resource
def load_model():
    model  = joblib.load('disease_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('diabetes.csv')
    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

try:
    model, scaler = load_model()
    df = load_data()
    model_loaded = True
except:
    model_loaded = False


# ── HEADER
st.markdown("""
<h1 style='text-align:center; color:#1B4F8A;'>🏥 Disease Prediction System</h1>
<p style='text-align:center; color:gray; font-size:16px;'>
AI-powered Diabetes Risk Prediction using Machine Learning
</p>
<hr style='border: 1px solid #1B4F8A;'>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found! Please upload `disease_prediction_model.pkl` and `scaler.pkl`")
    st.info("💡 Run the Colab notebook first to generate and download the model files.")
    st.stop()


# ── TABS 
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Data Insights", "ℹ️ About"])


# TAB 1 — PREDICT

with tab1:
    st.markdown("### 👤 Enter Patient Details")
    st.markdown("Adjust the values below and click **Predict** to get the result.")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("🤰 Pregnancies",
                        min_value=0, max_value=20, value=1, step=1)
        glucose     = st.slider("🍬 Glucose Level",
                        min_value=50, max_value=250, value=120)
        blood_pressure = st.slider("💉 Blood Pressure",
                        min_value=30, max_value=130, value=75)
        skin_thickness = st.slider("📏 Skin Thickness",
                        min_value=5, max_value=100, value=25)

    with col2:
        insulin     = st.slider("💊 Insulin Level",
                        min_value=10, max_value=900, value=80)
        bmi         = st.slider("⚖️ BMI",
                        min_value=10.0, max_value=70.0, value=28.0, step=0.1)
        dpf         = st.slider("🧬 Diabetes Pedigree Function",
                        min_value=0.05, max_value=2.5, value=0.5, step=0.01)
        age         = st.slider("🎂 Age",
                        min_value=18, max_value=90, value=30)

    st.markdown("---")

    # ── PREDICT BUTTON
    if st.button("🔍 Predict Diabetes Risk"):
        patient = {
            'Pregnancies':              pregnancies,
            'Glucose':                  glucose,
            'BloodPressure':            blood_pressure,
            'SkinThickness':            skin_thickness,
            'Insulin':                  insulin,
            'BMI':                      bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age':                      age
        }

        input_df = pd.DataFrame([patient])
        input_sc = scaler.transform(input_df)
        pred     = model.predict(input_sc)[0]
        prob     = model.predict_proba(input_sc)[0][1] * 100

        # ── RESULT
        if pred == 1:
            st.markdown(f"""
            <div class='result-box diabetic'>
                🔴 HIGH DIABETES RISK DETECTED<br>
                <span style='font-size:18px;'>Risk Probability: {prob:.1f}%</span>
            </div>""", unsafe_allow_html=True)
            st.warning("⚠️ Please consult a doctor for proper medical advice.")
        else:
            st.markdown(f"""
            <div class='result-box not-diabetic'>
                🟢 LOW DIABETES RISK<br>
                <span style='font-size:18px;'>Risk Probability: {prob:.1f}%</span>
            </div>""", unsafe_allow_html=True)
            st.success("✅ Keep maintaining a healthy lifestyle!")

        # ── PROBABILITY GAUGE
        st.markdown("#### 📊 Risk Probability Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🔴 Diabetic Risk",     f"{prob:.1f}%")
        with col_b:
            st.metric("🟢 Non-Diabetic Risk", f"{100-prob:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(['Risk'], [prob],        color='#E74C3C', height=0.4, label='Diabetic')
        ax.barh(['Risk'], [100 - prob],  color='#27AE60', height=0.4,
                left=[prob], label='Not Diabetic')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability %')
        ax.legend(loc='upper right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # ── INPUT SUMMARY 
        st.markdown("#### 📋 Patient Input Summary")
        summary_df = pd.DataFrame([patient]).T
        summary_df.columns = ['Value']
        st.dataframe(summary_df, use_container_width=True)

        # ── HEALTH TIPS 
        st.markdown("#### 💡 Health Recommendations")
        if glucose > 140:
            st.warning("🍬 Glucose is HIGH — reduce sugar intake")
        if bmi > 30:
            st.warning("⚖️ BMI is HIGH — consider regular exercise")
        if blood_pressure > 90:
            st.warning("💉 Blood Pressure is HIGH — reduce salt intake")
        if prob < 40:
            st.info("✅ Your vitals look healthy — keep it up!")



# TAB 2 — DATA INSIGHTS

with tab2:
    st.markdown("### 📊 Dataset Insights")

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients",   len(df))
    col2.metric("Diabetic",         int(df['Outcome'].sum()))
    col3.metric("Non-Diabetic",     int(len(df) - df['Outcome'].sum()))
    col4.metric("Features",         len(df.columns) - 1)

    st.markdown("---")

    # Outcome pie chart
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🥧 Outcome Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['Outcome'].value_counts()
        ax.pie(counts, labels=['Not Diabetic', 'Diabetic'],
               autopct='%1.1f%%', startangle=90,
               colors=['#27AE60', '#E74C3C'],
               explode=(0, 0.05))
        ax.set_title('Patient Outcome Split')
        st.pyplot(fig)

    with col_r:
        st.markdown("#### 📈 Feature Distributions")
        feature_choice = st.selectbox("Select Feature",
                            df.columns[:-1].tolist())
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df[df['Outcome']==0][feature_choice],
                bins=20, alpha=0.7, color='#27AE60', label='Not Diabetic')
        ax.hist(df[df['Outcome']==1][feature_choice],
                bins=20, alpha=0.7, color='#E74C3C', label='Diabetic')
        ax.set_title(f'{feature_choice} Distribution')
        ax.set_xlabel(feature_choice)
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)

    # Correlation heatmap
    st.markdown("#### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt='.2f',
                cmap='Blues', ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlations')
    st.pyplot(fig)

    # Raw data
    st.markdown("#### 📄 Raw Dataset")
    st.dataframe(df.head(20), use_container_width=True)



# TAB 3 — ABOUT
with tab3:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **Disease Prediction System** is an AI-powered web application that predicts
    the risk of diabetes using Machine Learning.

    ---

    #### 🛠️ Tech Stack
    - **Python** — Core programming language
    - **Scikit-learn** — Machine Learning models
    - **Pandas & NumPy** — Data processing
    - **Streamlit** — Web application
    - **Matplotlib & Seaborn** — Visualizations

    ---

    #### 📊 Dataset
    - **Pima Indians Diabetes Database** (Kaggle)
    - 768 patients, 8 features
    - Binary classification (Diabetic / Not Diabetic)

    ---

    #### 🤖 Model
    - **Algorithm:** Random Forest Classifier
    - **Accuracy:** ~77.92%
    - **Train/Test Split:** 80% / 20%

    ---

    #### 👨‍💻 Developer
    **Veerendra Challa**
    B.Tech CSE (AI & ML) — Uttaranchal University
    """)


# ── FOOTER 
st.markdown("""
<hr>
<p style='text-align:center; color:gray; font-size:13px;'>
⚠️ This app is for educational purposes only. Not a substitute for medical advice.<br>
Built by <b>Veerendra Challa</b> | Powered by Streamlit & Scikit-learn
</p>
""", unsafe_allow_html=True)
