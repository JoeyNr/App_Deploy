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
# ENCODING MAPS
# ------------------------------------------------
city_map = {
    'Dakar': 0, 'Saint-Louis': 1, 'Tambacounda': 2, 'Fatick': 3, 'Thies': 4,
    'Louga': 5, 'Kaffrine': 6, 'Diourbel': 7, 'Kolda': 8, 'Matam': 9,
    'Sedhiou': 10, 'Kaolack': 11, 'Ziguinchor': 12, 'Kedougou': 13
}

duration_map = {
    '24+': 0, '15 - 18': 1, '12 - 15': 2, '21 - 24': 3, '18 - 21': 4,
    '6 - 9': 5, '9 - 12': 6, '3 - 6': 7
}

top_pack_list = [
    'All-net 500F=2000F', '5d, Data: 100 F=40MB,24H', 'All-net 500F=2000F;5d',
    'On net 200F=Unlimited _call24H', 'Data:490F=1GB,7d', 'Data:1000F=5GB,7d',
    'VAS(IVR_Radio_Daily)', 'Data:200F=Unlimited,24H', 'Jokko_Daily',
    'Mixt 250F=Unlimited_call24H', 'Data: 200 F=100MB,24H',
    'MIXT:500F= 2500F on net _2500F off net;2d', 'Data:1000F=2GB,30d',
    'IVR Echat_Daily_50F', 'On-net 1000F=10MilF;10d', 'All-net 600F= 3000F ;5d',
    'Twter_U2opia_Daily', 'MIXT: 200mnoff net _unl on net _5Go;30d',
    'On-net 500F_FNF;3d', 'Twter_U2opia_Weekly', 'All-net 500F =2000F_AllNet_Unlimited',
    'Yewouleen_PKG', 'On-net 500=4000,10d', 'On-net 200F=60mn;1d',
    'Data:3000F=10GB,30d', 'Incoming_Bonus_woma', 'All-net 1000=5000;5d',
    'Data:500F=2GB,24H', 'Data:300F=100MB,2d', 'MIXT: 390F=04HOn-net_400SMS_400 Mo;4h',
    'Data:50F=30MB_24H', 'All-net 1000F=(3000F On+3000F Off);5d',
    'All-net 500F=1250F_AllNet_1250_Onnet;48h', 'Data:150F=SPPackage1,24H',
    'CVM_on-net bundle 500=5000', 'Data: 200F=1GB,24H', '200=Unlimited1Day',
    'MROMO_TIMWES_OneDAY', '200F=10mnOnNetValid1H',
    'On net 200F= 3000F_10Mo ;24H', 'All-net 300=600;2d', 'On-net 300F=1800F;3d',
    'MIXT: 590F=02H_On-net_200SMS_200 Mo;24h', 'YMGX 100=1 hour FNF, 24H/1 month',
    '500=Unlimited3Day', 'Data:DailyCycle_Pilot_1.5GB', 'Data:1500F=3GB,30D'
    # Add more if needed...
]
top_pack_map = {name: idx for idx, name in enumerate(top_pack_list)}

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
    st.title("\U0001F4E1 Welcome to the Telecom Churn Predictor")
    st.write("This app predicts whether a telecom customer is likely to churn based on key usage metrics.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("\U0001F4C8 Predict Customer Churn")
    st.write("Enter customer details below:")

    with st.form("prediction_form"):
        city = st.selectbox("City", list(city_map.keys()))
        duration = st.selectbox("Duration (Months)", list(duration_map.keys()))
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
        top_pack = st.selectbox("Top Pack", top_pack_list)
        freq_top_pack = st.number_input("Top Pack Frequency", min_value=0.0, step=1.0)

        submit = st.form_submit_button("Predict Churn")

    if submit:
        encoded_city = city_map.get(city, -1)
        encoded_duration = duration_map.get(duration, -1)
        encoded_top_pack = top_pack_map.get(top_pack, -1)

        if -1 in (encoded_city, encoded_duration, encoded_top_pack):
            st.error("⚠️ Encoding failed for one of the categorical values.")
        else:
            input_df = pd.DataFrame([[
                encoded_city, encoded_duration, recharge_amt, recharge_freq,
                total_revenue, avg_revenue, frequency, data_volume,
                on_net, orange, tigo, days_active, encoded_top_pack, freq_top_pack
            ]], columns=X_columns)

            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            st.markdown("### \U0001F9FE Prediction Result")
            st.success("✅ Customer is likely to churn" if prediction == 1 else "❌ Customer will not churn")
            st.metric("Churn Probability", f"{probability:.2%}")
            # ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("\U0001F4D8 About This App")
    st.markdown("""
        *Group 4 - Capstone Project (Telecom Customer Churn)*

        - *Model:* XGBoost (after sampling)  
        - *Accuracy:* 86%  
        - *Dataset:* Expresso Senegal  
        - *Input Features:* Recharge behavior, call volume, revenue, and activity levels  

        *Built with ❤️ using:* Python, Streamlit, Scikit-learn, XGBoost
    """)
