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
