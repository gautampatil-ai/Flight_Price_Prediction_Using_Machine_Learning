"""
Flight Price Prediction App
-----------------------------
A Streamlit interface for an XGBoost Regressor that predicts flight
ticket prices based on trip details.

Run locally with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "flight_price_model.pkl"

# Feature order the model was trained on (extracted from the model file)
FEATURE_ORDER = [
    "Unnamed: 0", "flight", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class", "duration", "days_left",
]

# Label-encoding maps, based on the standard alphabetical ordering
# scikit-learn's LabelEncoder produces for this well-known flight-price
# dataset schema. VERIFY these against your own training notebook /
# saved encoder before relying on this app for real predictions.
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
CITY_MAP = {c: i for i, c in enumerate(CITIES)}

TIME_SLOTS = ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"]
TIME_MAP = {t: i for i, t in enumerate(TIME_SLOTS)}

STOPS = ["one", "two_or_more", "zero"]
STOPS_MAP = {s: i for i, s in enumerate(STOPS)}
STOPS_DISPLAY = {"zero": "Non-stop", "one": "1 Stop", "two_or_more": "2+ Stops"}

CLASSES = ["Business", "Economy"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------
# Cached model loader
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import pickle
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def build_feature_row(source_city, destination_city, departure_time, arrival_time,
                       stops, travel_class, duration, days_left):
    """Assemble a single-row DataFrame matching the model's expected feature order."""
    row = {
        "Unnamed: 0": 0,  # unused index artifact from the original training data
        "flight": 0,      # flight-number code unavailable without the original encoder
        "source_city": CITY_MAP[source_city],
        "departure_time": TIME_MAP[departure_time],
        "stops": STOPS_MAP[stops],
        "arrival_time": TIME_MAP[arrival_time],
        "destination_city": CITY_MAP[destination_city],
        "class": CLASS_MAP[travel_class],
        "duration": duration,
        "days_left": days_left,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("✈️ Flight Price Predictor")
    st.markdown(
        """
        Predicts flight ticket price using an **XGBoost Regressor**
        trained on flight-route and booking details.

        **How to use:**
        1. Fill in the trip details.
        2. Click **Predict Price**.
        3. View the estimated fare.
        """
    )
    st.divider()
    st.warning(
        "⚠️ **Encoding assumption notice**\n\n"
        "This model expects categorical fields (city, time, stops, class) "
        "as label-encoded numbers. No encoder file was provided, so this "
        "app uses standard alphabetical label-encoding as an assumption. "
        "If your training notebook used a different encoding order, "
        "predictions will be inaccurate — replace the mapping dictionaries "
        "at the top of `app.py` with your actual saved encoder mappings.\n\n"
        "The `flight` (flight number) and `Unnamed: 0` (row index) fields "
        "are also set to a fixed placeholder value, since no encoder for "
        "flight numbers was provided.",
        icon="⚠️",
    )
    st.divider()
    st.caption("Built with Streamlit · XGBoost · pandas")


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Make sure `flight_price_model.pkl` is in "
        "the same directory as `app.py`."
    )
    st.stop()
except ModuleNotFoundError:
    st.error(
        "The `xgboost` package is required to load this model. "
        "Make sure it's listed in `requirements.txt`."
    )
    st.stop()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("✈️ Flight Price Predictor")
st.markdown("Enter your trip details below to estimate the ticket price.")

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
st.subheader("Trip Details")

col1, col2 = st.columns(2)
with col1:
    source_city = st.selectbox("Source City", CITIES, index=2)
with col2:
    destination_city = st.selectbox("Destination City", CITIES, index=5)

col3, col4 = st.columns(2)
with col3:
    departure_time = st.selectbox("Departure Time", TIME_SLOTS, index=4)
with col4:
    arrival_time = st.selectbox("Arrival Time", TIME_SLOTS, index=2)

col5, col6 = st.columns(2)
with col5:
    stops = st.selectbox(
        "Number of Stops", STOPS,
        format_func=lambda s: STOPS_DISPLAY[s],
        index=2,
    )
with col6:
    travel_class = st.selectbox("Class", CLASSES, index=1)

col7, col8 = st.columns(2)
with col7:
    duration = st.number_input(
        "Flight Duration (hours)", min_value=0.5, max_value=48.0,
        value=2.5, step=0.25,
    )
with col8:
    days_left = st.number_input(
        "Days Left Until Departure", min_value=0, max_value=365, value=15, step=1,
    )

if source_city == destination_city:
    st.warning("Source and destination cities are the same — please pick different cities.")

st.divider()
predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if predict_clicked:
    if source_city == destination_city:
        st.error("Please choose different source and destination cities.")
    else:
        with st.spinner("Predicting..."):
            X = build_feature_row(
                source_city, destination_city, departure_time, arrival_time,
                stops, travel_class, duration, days_left,
            )
            prediction = model.predict(X)[0]

        st.subheader("Estimated Price")
        st.success(f"### ₹ {prediction:,.0f}")
        st.caption(
            "This is an estimate based on the model's training data and "
            "the encoding assumptions noted in the sidebar. Actual fares "
            "may vary."
        )

        with st.expander("View input features sent to the model"):
            st.dataframe(X, use_container_width=True)

st.divider()
st.caption(
    "Note: Predictions are generated by a machine learning model and may "
    "not reflect real-time airfare. Use as guidance only."
)
