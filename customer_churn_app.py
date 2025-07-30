import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# ------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #E3E7F7 !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            border-right: none;
        }

        html, body, [data-testid="stAppViewContainer"] > .main {
            background-color: white !important;
            color: black !important;
        }

        .stSelectbox > div,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #E3E7F7 !important;
            border-radius: 8px;
        }

        input[type="number"] {
            background-color: #E3E7F7 !important;
            border-radius: 8px;
            padding: 0.4rem;
        }

        div[data-baseweb="slider"] > div > div > div:nth-child(2) {
            background: #4B0082 !important;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(3) {
            background: #e6e6e6 !important;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #4B0082 !important;
        }

        div.stButton > button {
            background-color: #4B0082 !important;
            color: white !important;
            border-radius: 8px !important;
            height: 3em;
            width: auto;
            padding: 0.6rem 1.5rem;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #3a006b !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL ARTIFACTS
# ------------------------------------------------
model = joblib.load('g4_xgb_model.pkl')
scaler = joblib.load('g4_scaler.pkl')
X_columns = joblib.load('g4_column_names.pkl')

# ------------------------------------------------
# SIDEBAR MENU
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#E3E7F7"},
            "icon": {"color": "#4B0082", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#D5D9F0",
                "color": "#333333"
            },
            "nav-link-selected": {
                "background-color": "#C2C7EA",
                "color": "#000000"
            },
        },
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("📡 Welcome to the Telecom Churn Predictor")
    st.write("This app predicts whether a telecom customer is likely to churn based on key usage metrics.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("📈 Predict Customer Churn")
    st.write("Enter customer details below:")

    with st.form("prediction_form"):
        city = st.selectbox("City (Encoded)", list(range(0, 21)), index=5)
        duration = st.selectbox("Duration (Months) (Encoded)", list(range(0, 8)), index=4)
        recharge_amt = st.number_input("Recharge Amount (XOF)", min_value=0.0, step=100.0)
        recharge_freq = st.number_input("Recharge Frequency", min_value=0.0, step=1.0)
        total_revenue = st.number_input("Total Revenue (XOF)", min_value=0.0, step=100.0)
        avg_revenue = st.number_input("Average Revenue (XOF)", min_value=0.0, step=100.0)
        frequency = st.number_input("Frequency", min_value=0.0, step=1.0)
        data_volume = st.number_input("Data Volume", min_value=0.0, step=100.0)
        on_net = st.number_input("On-Net Calls", min_value=0.0, step=1.0)
        orange = st.number_input("Orange Calls", min_value=0.0, step=1.0)
        tigo = st.number_input("Tigo Calls", min_value=0.0, step=1.0)
        days_active = st.number_input("Days Active", min_value=0, step=1)
        top_pack = st.selectbox("Top Pack (Encoded)", list(range(0, 21)), index=5)
        freq_top_pack = st.number_input("Top Pack Frequency", min_value=0.0, step=1.0)

        submit = st.form_submit_button("Predict Churn")

    if submit:
        input_df = pd.DataFrame([[
            city, duration, recharge_amt, recharge_freq, total_revenue, avg_revenue,
            frequency, data_volume, on_net, orange, tigo, days_active, top_pack, freq_top_pack
        ]], columns=X_columns)

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        st.markdown("### 🧾 Prediction Result")
        st.success("✅ Customer is likely to churn" if prediction == 1 else "❌ Customer will not churn")
        st.metric("Churn Probability", f"{probability:.2%}")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("📘 About This App")
    st.markdown("""
        *Group 4 - Capstone Project (Telecom Customer Churn)*

        - *Model:* XGBoost (after sampling)  
        - *Accuracy:* 86%  
        - *Dataset:* Expresso Senegal  
        - *Input Features:* Recharge behavior, call volume, revenue, and activity levels  
        
        *Built with ❤️ using:* Python, Streamlit, Scikit-learn, XGBoost
    """)
