"""
Flight Price Prediction App
-----------------------------
Run locally with:
    streamlit run app.py
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

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

FEATURE_ORDER = [
    "Unnamed: 0", "flight", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class", "duration", "days_left",
]

CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
CITY_MAP = {c: i for i, c in enumerate(CITIES)}

TIME_SLOTS = ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"]
TIME_MAP = {t: i for i, t in enumerate(TIME_SLOTS)}

STOPS = ["one", "two_or_more", "zero"]
STOPS_MAP = {s: i for i, s in enumerate(STOPS)}
STOPS_DISPLAY = {"zero": "Non-stop", "one": "1 Stop", "two_or_more": "2+ Stops"}

CLASSES = ["Business", "Economy"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_feature_row(source_city, destination_city, departure_time, arrival_time,
                       stops, travel_class, duration, days_left):
    row = {
        "Unnamed: 0": 0,
        "flight": 0,
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


with st.sidebar:
    st.title("✈️ Flight Price Predictor")
    st.markdown(
        """
        Estimate flight ticket prices based on route, timing,
        stops, class, and how far in advance you book.
        """
    )
    st.divider()
    st.caption("Built with Streamlit · XGBoost")

model = load_model()

st.title("✈️ Flight Price Predictor")
st.markdown("Enter your trip details to get an estimated ticket price.")

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

st.divider()
predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

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
