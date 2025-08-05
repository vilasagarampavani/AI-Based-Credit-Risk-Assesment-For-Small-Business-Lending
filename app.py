import streamlit as st
import pandas as pd
import numpy as np
import joblib
# --- 🔹 Load Model, Scaler & Features ---
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("model_features.pkl")
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Assessment Tool")
st.markdown("Predict whether a small business loan is likely to **default** based on SBA data.")
# --- 🔹 Input Form ---
st.sidebar.header("📋 Borrower & Loan Details")
with st.sidebar.form("input_form"):
    Term = st.number_input("Loan Term (in months)", min_value=1, value=60)
    NoEmp = st.number_input("Number of Employees", min_value=0, value=10)
    NewExist = st.selectbox("Business Type", options=[1, 2], format_func=lambda x: "New" if x == 1 else "Existing")
    CreateJob = st.number_input("Jobs Created", min_value=0, value=1)
    RetainedJob = st.number_input("Jobs Retained", min_value=0, value=1)
    FranchiseCode = st.number_input("Franchise Code (0 = None)", min_value=0, value=0)
    RevLineCr = st.selectbox("Revolving Line of Credit?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    LowDoc = st.selectbox("Low Documentation?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    DisbursementGross = st.number_input("Disbursement Gross ($)", min_value=0.0, value=50000.0)
    BalanceGross = st.number_input("Balance Gross ($)", min_value=0.0, value=0.0)
    ChgOffPrinGr = st.number_input("Charged Off Principal ($)", min_value=0.0, value=0.0)
    GrAppv = st.number_input("Gross Approval Amount ($)", min_value=0.0, value=50000.0)
    submit = st.form_submit_button("🚀 Predict Risk")
# --- 🔹 Make Prediction ---
if submit:
    try:
        input_data = pd.DataFrame([[
            Term, NoEmp, NewExist, CreateJob, RetainedJob, FranchiseCode, RevLineCr, LowDoc,
            DisbursementGross, BalanceGross, ChgOffPrinGr, GrAppv
        ]], columns=features)
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]
        if prediction == 1:
            st.error(f"❌ **High Risk of Default**\n\nProbability: {prob:.2%}")
        else:
            st.success(f"✅ **Low Risk of Default**\n\nProbability: {prob:.2%}")
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {str(e)}")
