import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📊 Telecom Customer Churn Prediction")
st.write("Fill in the customer details and click **Predict Churn**.")

# Load trained model pipeline
@st.cache_resource
def load_model():
    clf = joblib.load("churn_rf_pipeline.pkl")
    return clf

clf = load_model()

# --- Input fields based on Telco Customer Churn dataset ---

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen_flag = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

with col2:
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

col3, col4 = st.columns(2)

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col4:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=2000.0)

# Build dataframe exactly like training features (Telco dataset)
data = {
    "gender": gender,
    "SeniorCitizen": 1 if SeniorCitizen_flag == "Yes" else 0,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

input_df = pd.DataFrame([data])

st.markdown("### 🔎 Preview of input data")
st.dataframe(input_df)

if st.button("🚀 Predict Churn"):
    try:
        pred = clf.predict(input_df)[0]
        prob = clf.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ Customer is likely to **CHURN**.\n\nChurn probability: **{prob:.2f}**")
        else:
            st.success(f"✅ Customer is **NOT** likely to churn.\n\nChurn probability: **{prob:.2f}**")
    except Exception as e:
        st.error("Something went wrong while making a prediction.")
        st.code(str(e))
