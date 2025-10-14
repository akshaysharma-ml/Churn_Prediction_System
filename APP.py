import streamlit as st
import pandas as pd
import pickle
import io

# ---------------------------------------------------------
# Load pre-trained model and scaler
# ---------------------------------------------------------
with open(r"C:\Users\hp\Desktop\ml_mini_project\CHURN.PY\logistic_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\hp\Desktop\ml_mini_project\CHURN.PY\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

st.markdown("""
Upload your **customer dataset (CSV or Excel)** and this app will predict  
which customers are likely to churn (leave the service).
""")

# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.subheader("✅ Uploaded Data Preview")
    st.dataframe(data.head())

    # Check if 'Churn' exists — if yes, drop it before prediction
    if "Churn" in data.columns:
        data = data.drop("Churn", axis=1)

    # ---------------------------------------------------------
    # Data preprocessing (must match training structure)
    # ---------------------------------------------------------
    # Encode categorical values like during training
    # Simple approach: convert 'Yes'/'No' and similar strings
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("category").cat.codes

    # Scale features
    data_scaled = scaler.transform(data)

    # ---------------------------------------------------------
    # Predict churn
    # ---------------------------------------------------------
    predictions = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)[:, 1]

    # Add results to dataframe
    result_df = data.copy()
    result_df["Churn_Predicted"] = predictions
    result_df["Churn_Probability"] = probabilities

    st.subheader("🔮 Prediction Results")
    st.dataframe(result_df.head())

    # ---------------------------------------------------------
    # Download predictions
    # ---------------------------------------------------------
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload a CSV or Excel file to start predictions.")
