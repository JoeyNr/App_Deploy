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
        city = st.selectbox("City", ['Dakar','Saint-Louis','Tambacounda','Fatick','Thies', 'Louga', 'Kaffrine', 'Diourbel', 'Kolda','Matam', 'Sedhiou', 'Kaolack', 'Ziguinchor', 'Kedougou'])
        duration = st.selectbox("Duration (Months)", ['24+', '15 - 18', '12 - 15', '21 - 24', '18 - 21', '6 - 9', '9 - 12', '3 - 6'])
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
        top_pack = st.selectbox("Top Pack", ['All-net 500F=2000F','5d, Data: 100 F=40MB,24H','All-net 500F=2000F;5d','On net 200F=Unlimited _call24H',
        'Data:490F=1GB,7d','Data:1000F=5GB,7d','VAS(IVR_Radio_Daily)','Data:200F=Unlimited,24H','Jokko_Daily',
        'Mixt 250F=Unlimited_call24H','Data: 200 F=100MB,24H','MIXT:500F= 2500F on net _2500F off net;2d',
        'Data:1000F=2GB,30d','IVR Echat_Daily_50F','On-net 1000F=10MilF;10d','All-net 600F= 3000F ;5d',
        'Twter_U2opia_Daily','MIXT: 200mnoff net _unl on net _5Go;30d','On-net 500F_FNF;3d','Twter_U2opia_Weekly',
        'All-net 500F =2000F_AllNet_Unlimited','Yewouleen_PKG','On-net 500=4000,10d','On-net 200F=60mn;1d',
        'Data:3000F=10GB,30d','Incoming_Bonus_woma','All-net 1000=5000;5d','Data:500F=2GB,24H','Data:300F=100MB,2d',
        'MIXT: 390F=04HOn-net_400SMS_400 Mo;4h','Data:50F=30MB_24H','All-net 1000F=(3000F On+3000F Off);5d',
        'All-net 500F=1250F_AllNet_1250_Onnet;48h','Data:150F=SPPackage1,24H','CVM_on-net bundle 500=5000',
        'Data: 200F=1GB,24H','200=Unlimited1Day','MROMO_TIMWES_OneDAY','200F=10mnOnNetValid1H','On net 200F= 3000F_10Mo ;24H',
        'All-net 300=600;2d','On-net 300F=1800F;3d','MIXT: 590F=02H_On-net_200SMS_200 Mo;24h',
        'YMGX 100=1 hour FNF, 24H/1 month','500=Unlimited3Day','Data:DailyCycle_Pilot_1.5GB','Data:1500F=3GB,30D',
        'New_YAKALMA_4_ALL','Twter_U2opia_Monthly','Jokko_promo','Pilot_Youth4_490','SUPERMAGIK_5000',
        'On-net 2000f_One_Month_100H; 30d','Jokko_Monthly','Jokko_Weekly','Facebook_MIX_2D','Data: 490F=Night,00H-08H',
        'Internat: 1000F_Zone_1;24H','FNF2 ( JAPPANTE)','All-net 500F=4000F ; 5d','WIFI_Family_2MBPS','Data:700F=SPPackage1,7d',
        'Data:700F=1.5GB,7d','Data:30Go_V 30_Days','DataPack_Incoming','MIXT: 500F=75(SMS, ONNET, Mo)_1000FAllNet;24h',
        'EVC_500=2000F','MIXT:1000F=4250 Off net _ 4250F On net _100Mo; 5d','MROMO_TIMWES_RENEW','All-net 5000= 20000off+20000on;30d',
        'Pilot_Youth1_290','Data:1500F=SPPackage1,30d','MIXT:10000F=10hAllnet_3Go_1h_Zone3;30d','EVC_Jokko_Weekly',
        'MIXT: 5000F=80Konnet_20Koffnet_250Mo;30d','WIFI_ Family _4MBPS','Internat: 1000F_Zone_3;24h',
        'Mixt : 500F=2500Fonnet_2500Foffnet ;5d','FIFA_TS_daily','CVM_100F_unlimited','WIFI_ Family _10MBPS','301765007',
        'SUPERMAGIK_1000','TelmunCRBT_daily','pilot_offer6','305155009','VAS(IVR_Radio_Monthly)','Staff_CPE_Rent',
        'IVR Echat_Weekly_200F','1000=Unlimited7Day','FNF_Youth_ESN','EVC_100Mo','Data:New-GPRS_PKG_1500F',
        'Data_EVC_2Go24H','CVM_100f=200 MB','Internat: 2000F_Zone_2;24H','MIXT: 4900F= 10H on net_1,5Go ;30d',
        'VAS(IVR_Radio_Weekly)','pilot_offer5','CVM_200f=400MB','APANews_weekly','CVM_500f=2GB',
        'CVM_On-net 1300f=12500','pack_chinguitel_24h','NEW_CLIR_TEMPALLOWED_LIBERTE_MOBILE',
        'NEW_CLIR_PERMANENT_LIBERTE_MOBILE','GPRS_3000Equal10GPORTAL','APANews_monthly','200=unlimited pilot auto',
        'IVR Echat_Monthly_500F','CVM_On-net 400f=2200F','EVC_MEGA10000F'])
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
