import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Car Price Prediction App
# -----------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to estimate its price.")

MODEL_PATH = Path("model.joblib")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    model = None
    st.error(f"Could not load the model: {error}")

if model is None:
    st.warning(
        "The model file was not found. Add your trained model to this "
        "repository and name it exactly: model.joblib"
    )
    st.stop()

# Use the exact input column names saved by many scikit-learn models.
feature_names = getattr(model, "feature_names_in_", None)

if feature_names is None:
    st.error(
        "This model does not expose its training column names. "
        "Please check the model-training notebook and add the exact "
        "input columns to this app."
    )
    st.stop()

st.subheader("Car details")
user_values = {}

with st.form("car_details_form"):
    for feature in feature_names:
        value = st.text_input(
            str(feature).replace("_", " ").title(),
            placeholder=f"Enter {feature}"
        )
        user_values[feature] = value.strip()

    submitted = st.form_submit_button("Predict Price")

if submitted:
    empty_fields = [name for name, value in user_values.items() if value == ""]

    if empty_fields:
        st.error("Please fill in all fields before predicting.")
    else:
        # Convert values that look numeric; keep other values as text.
        row = {}
        for name, value in user_values.items():
            try:
                row[name] = float(value)
            except ValueError:
                row[name] = value

        input_data = pd.DataFrame([row], columns=list(feature_names))

        try:
            prediction = model.predict(input_data)
            predicted_price = float(prediction[0])

            st.success("Prediction completed!")
            st.metric("Estimated Car Price", f"{predicted_price:,.2f}")
            st.caption(
                "This is an estimate from the trained model, not a guaranteed market price."
            )
        except Exception as error:
            st.error(
                "The model could not process these values. Check that each "
                "field uses the same format and categories as the training data."
            )
            st.code(str(error))

st.divider()
st.caption("Car Price Prediction • Built with Streamlit")
