import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")


model = load_model()
FEATURES = list(model.feature_names_in_)  # exact column order the model expects


def options_for(prefix):
    """Read category names straight from the model's one-hot columns."""
    return sorted(c.replace(prefix, "") for c in FEATURES if c.startswith(prefix))


MAKES = options_for("make_")
SIZES = options_for("vehicle size_")
TRANSMISSIONS = options_for("transmission type_")


def build_input(year, hp, make, size, transmission):
    """Create a 1-row DataFrame matching the training columns (one-hot encoded)."""
    row = pd.DataFrame(0, index=[0], columns=FEATURES)
    row.loc[0, "year"] = year
    row.loc[0, "engine hp"] = hp
    row.loc[0, f"make_{make}"] = 1
    row.loc[0, f"vehicle size_{size}"] = 1
    row.loc[0, f"transmission type_{transmission}"] = 1
    return row


# ---------------- UI ----------------
st.title("🚗 Car Price Predictor")
st.write("Enter the car's details and get an estimated price.")

col1, col2 = st.columns(2)
with col1:
    make = st.selectbox("Make", MAKES, index=MAKES.index("Toyota") if "Toyota" in MAKES else 0)
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015, step=1)
    hp = st.number_input("Engine HP", min_value=50, max_value=1200, value=200, step=5)
with col2:
    size = st.selectbox("Vehicle size", SIZES)
    transmission = st.selectbox("Transmission type", TRANSMISSIONS)

if st.button("Predict price", type="primary", use_container_width=True):
    X = build_input(year, hp, make, size, transmission)
    raw = float(model.predict(X)[0])

    # The model's raw output (~10-11) looks like a LOG-transformed price.
    # If you trained on log1p(price) use expm1; if you trained on raw price, use `price = raw`.
    price = float(np.expm1(raw))

    st.success(f"Estimated price: **${price:,.0f}**")
    with st.expander("Show model input"):
        st.dataframe(X.loc[:, (X != 0).any()])
