
import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #0E1726;
    color: white;
}
h1 {
    color: #00D4FF;
    text-align: center;
}
h3 {
    color: #00D4FF;
}
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model.pkl"
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found or loading failed: {e}")
    st.stop()

# Heading
st.title("💳 Online Payment Fraud Detection")

st.markdown(
    "<p style='text-align:center;'>"
    "Machine Learning Based Binary Classification System"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# Input Section
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:

    step = st.number_input(
        "Transaction Step",
        min_value=1,
        value=100
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=50000.0
    )

    oldbalanceOrg = st.number_input(
        "Old Balance Origin",
        min_value=0.0,
        value=50000.0
    )

with col2:

    newbalanceOrig = st.number_input(
        "New Balance Origin",
        min_value=0.0,
        value=0.0
    )

    oldbalanceDest = st.number_input(
        "Old Balance Destination",
        min_value=0.0,
        value=1000.0
    )

    newbalanceDest = st.number_input(
        "New Balance Destination",
        min_value=0.0,
        value=51000.0
    )

st.divider()

# Prediction
if st.button("🔍 Detect Fraud", use_container_width=True):

    input_data = pd.DataFrame({
        "step": [step],
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    try:
        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Transaction Prediction")

        if prediction == 1:
            st.error("🚨 FRAUD TRANSACTION DETECTED")

        else:
            st.success("✅ NOT A FRAUD TRANSACTION")

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.subheader("Transaction Information")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()

st.caption(
    "Developed using Python | Pandas | Streamlit | Logistic Regression"
)

