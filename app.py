# app.py

import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load model + features with fallbacks and friendly errors
model = None
feature_columns = None
try:
    # preferential name used in the Streamlit app
    model = joblib.load("xgb_churn_model.pkl")
except Exception:
    try:
        # fallback to the model filename created by the notebook
        model = joblib.load("customer_churn_model.pkl")
    except Exception:
        # model not found; we'll surface a friendly message later in the UI
        model = None

try:
    feature_columns = joblib.load("feature_columns.pkl")
except Exception:
    try:
        feature_columns = joblib.load("model_features.pkl")
    except Exception:
        feature_columns = None

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
# Add a simple banner and improved styling
st.markdown(
    "<div style='background: linear-gradient(90deg,#4b6cb7,#182848); padding: 18px; border-radius:8px;'>"
    "<h1 style='color: white; margin: 0; font-family:Segoe UI, Roboto, sans-serif;'>📊 Customer Churn Prediction Dashboard</h1>"
    "<p style='color: #dbeafe; margin: 4px 0 0 0;'>Interactive dashboard with predictions and explainability</p>"
    "</div>",
    unsafe_allow_html=True,
)

# small custom CSS to improve look
st.markdown(
    "<style>"
    ".stButton>button{background:linear-gradient(90deg,#06b6d4,#3b82f6); color:white;}"
    ".kpi{padding:12px; border-radius:8px; background:linear-gradient(180deg,#ffffff,#f1f5f9); box-shadow: 0 2px 6px rgba(0,0,0,0.08);}"
    "</style>",
    unsafe_allow_html=True,
)

# Load data (for visuals)
@st.cache_data
def load_data():
    # CSV lives inside the data/ folder in this workspace
    data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data.drop("customerID", axis=1, inplace=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)
    return data

data = load_data()

# If feature columns weren't saved with the model, infer them from the data
if feature_columns is None:
    feature_columns = [c for c in data.columns if c != "Churn"]

# Build per-column LabelEncoders from the training data so we can transform
# user input the same way the model saw training data.
encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    # ensure consistent string dtype (handles NaN)
    encoders[col] = le.fit(data[col].astype(str))

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Predict Customer", "Explainability"])

if page == "Dashboard":
    churn_rate = data["Churn"].value_counts(normalize=True).get(1, 0) * 100
    total_customers = len(data)
    avg_tenure = data["tenure"].mean()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("<div class='kpi'><h3 style='margin:0'>Total Customers</h3><h2 style='margin:4px'>{:,}</h2></div>".format(total_customers), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi'><h3 style='margin:0'>Churn Rate</h3><h2 style='margin:4px'>{:.2f}%</h2></div>".format(churn_rate), unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi'><h3 style='margin:0'>Avg Tenure (months)</h3><h2 style='margin:4px'>{:.1f}</h2></div>".format(avg_tenure), unsafe_allow_html=True)

    st.write("### Customer Distribution & Tenure")
    left, right = st.columns([2,1])
    with left:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(x="Churn", data=data, palette="viridis", ax=ax)
        ax.set_title("Churn Distribution")
        st.pyplot(fig)
    with right:
        fig2, ax2 = plt.subplots(figsize=(4,4))
        sns.boxplot(x="Churn", y="tenure", data=data, ax=ax2, palette="viridis")
        ax2.set_title("Tenure by Churn")
        st.pyplot(fig2)


# 🔮 PREDICTION PAGE

elif page == "Predict Customer":
    st.subheader("� Predict Customer Churn")

    if model is None:
        st.warning("Model artifact not found. Train the model or provide the model pickle in the workspace (customer_churn_model.pkl or xgb_churn_model.pkl). Prediction is disabled.")
    else:
        with st.expander("Customer input form", expanded=True):
            # split into two columns for a cleaner form
            col_left, col_right = st.columns(2)
            input_data = {}
            for i, col in enumerate(feature_columns):
                target_col = col_left if i % 2 == 0 else col_right
                with target_col:
                    if col in data.columns and data[col].nunique() <= 8:
                        input_data[col] = st.selectbox(col, sorted(data[col].unique()), key=f"in_{col}")
                    else:
                        if col in data.columns:
                            lo = float(data[col].min())
                            hi = float(data[col].max())
                            mean = float(data[col].mean())
                        else:
                            lo, hi, mean = 0.0, 1.0, 0.0
                        input_data[col] = st.number_input(col, lo, hi, mean, key=f"in_{col}")

        input_df = pd.DataFrame([input_data])

        # Transform categorical columns using the encoders we built from training data
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # Convert any remaining columns to numeric and fill missing using training means
        for col in input_df.columns:
            if input_df[col].dtype == object:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
            if input_df[col].isna().any():
                if col in data.columns:
                    input_df[col].fillna(data[col].mean(), inplace=True)
                else:
                    input_df[col].fillna(0, inplace=True)

        # Align columns to the order expected by the model and add missing columns with 0
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        predict_col, explain_col = st.columns([1,1])
        with predict_col:
            if st.button("🔎 Predict"):
                try:
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
                    if pred == 1:
                        if prob is not None:
                            st.error(f"❌ Customer likely to churn (Probability: {prob:.2f})")
                        else:
                            st.error("❌ Customer likely to churn")
                    else:
                        if prob is not None:
                            st.success(f"✅ Customer likely to stay (Probability: {prob:.2f})")
                        else:
                            st.success("✅ Customer likely to stay")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with explain_col:
            st.info("Model loaded: {}".format("Yes" if model is not None else "No"))
            st.caption("Tip: save and load the preprocessing pipeline to guarantee identical transforms as training.")


elif page == "Explainability":
    st.subheader("💡 Model Explainability (SHAP)")

    if model is None:
        st.warning("No model loaded; explainability is disabled.")
    else:
        try:
            explainer = shap.Explainer(model)
            X_sample = data[feature_columns].sample(min(200, len(data)), random_state=42)
            shap_values = explainer(X_sample)

            st.write("### Global Feature Importance")
            fig1, ax1 = plt.subplots(figsize=(8,4))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig1)

            st.write("### Individual Prediction Explanation")
            idx = st.slider("Select Row", 0, len(X_sample)-1, 0)
            fig_force = shap.plots.force(shap_values[idx], matplotlib=True)
            st.pyplot(fig_force)
        except Exception as e:
            st.error(f"Explainability failed: {e}")

    # Diagnostics expander
    with st.expander("Diagnostics & Artifacts", expanded=False):
        st.write({
            "model_loaded": model is not None,
            "feature_columns_count": len(feature_columns) if feature_columns is not None else 0,
            "available_pickles": [f for f in ["customer_churn_model.pkl","xgb_churn_model.pkl","model_features.pkl","feature_columns.pkl","label_encoder.pkl"] if os.path.exists(f)]
        })



#wrote the basic code ,then copilot did the css.
#I think that is pretty evident .