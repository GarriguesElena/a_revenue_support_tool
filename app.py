import pickle
from pathlib import Path

import pycountry

import pandas as pd
import streamlit as st

from datetime import date, timedelta




# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Booking Risk Analyzer",
    page_icon="🏨",
    layout="centered"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

# Default threshold used to convert probabilities into classes.
# It can be changed later without retraining the model.
DECISION_THRESHOLD = 0.50

MODEL_PATH = Path("models/random_forest_pipeline.pkl")

COUNTRY_CODES = [
    'ABW', 'AGO', 'AIA', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM',
    'ATA', 'ATF', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA',
    'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BOL', 'BRA', 'BRB',
    'BWA', 'CAF', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'CN', 'COL',
    'COM', 'CPV', 'CRI', 'CUB', 'CYM', 'CYP', 'CZE', 'DEU', 'DJI',
    'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ESP', 'EST', 'ETH',
    'FIN', 'FJI', 'FRA', 'FRO', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA',
    'GIB', 'GLP', 'GNB', 'GRC', 'GTM', 'GUY', 'HKG', 'HND', 'HRV',
    'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR',
    'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KHM', 'KIR',
    'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBY', 'LCA', 'LIE', 'LKA',
    'LTU', 'LUX', 'LVA', 'MAC', 'MAR', 'MCO', 'MDG', 'MDV', 'MEX',
    'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MOZ', 'MRT', 'MUS', 'MWI',
    'MYS', 'MYT', 'NAM', 'NCL', 'NGA', 'NIC', 'NLD', 'NOR', 'NPL',
    'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW', 'POL', 'PRI',
    'PRT', 'PRY', 'PYF', 'QAT', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN',
    'SEN', 'SGP', 'SLE', 'SLV', 'SMR', 'SRB', 'STP', 'SUR', 'SVK',
    'SVN', 'SWE', 'SYC', 'SYR', 'TGO', 'THA', 'TJK', 'TMP', 'TUN',
    'TUR', 'TWN', 'TZA', 'UGA', 'UKR', 'UMI', 'URY', 'USA', 'UZB',
    'VEN', 'VGB', 'VNM', 'ZAF', 'ZMB', 'ZWE'
]

COUNTRY_MAPPING = {}

for code in COUNTRY_CODES:

    country = pycountry.countries.get(alpha_3=code)

    if country:
        COUNTRY_MAPPING[country.name] = code

COUNTRY_MAPPING["Unknown"] = "Unknown"

def country_code_to_flag(code):
    # ---------------------------------------------------------
    # Convert a two-letter country code into a flag emoji.
    # ---------------------------------------------------------

    return "".join(
        chr(ord(character) + 127397)
        for character in code
    )
COUNTRY_OPTIONS = {}

for country_name, alpha_3_code in COUNTRY_MAPPING.items():

    if alpha_3_code == "Unknown":
        COUNTRY_OPTIONS["🌍 Unknown"] = "Unknown"
        continue

    country_object = pycountry.countries.get(
        alpha_3=alpha_3_code
    )

    if country_object:
        flag = country_code_to_flag(
            country_object.alpha_2
        )

        display_name = f"{flag} {country_name}"

        COUNTRY_OPTIONS[display_name] = alpha_3_code


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model(model_path: Path):
    """
    Load the trained Machine Learning pipeline.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model


try:
    model = load_model(MODEL_PATH)

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()


# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------

st.title("🏨 Booking Risk Analyzer")

st.write(
    """
    This tool estimates the probability that a hotel booking will be
    cancelled using the information available at the time of reservation.

    The prediction is intended to support management decisions and should
    not replace professional judgement.
    """
)


# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------

st.subheader("Booking Information")

# Hotel information
st.markdown("### 🏨 Hotel and Arrival")

hotel = st.selectbox(
    "Hotel type",
    options=[
        "City Hotel",
        "Resort Hotel"
    ]
)

lead_time = st.slider(
    "Lead time (how many days in advance the reservation was made)",
    min_value=0,
    max_value=1000,
    value=30,
    step=1
)

col1, col2 = st.columns(2)

with col1:
    check_in = st.date_input(
        "Check In",
        value=date.today()
    )

with col2:
    check_out = st.date_input(
        "Check Out (must be after Check In date)",
        value=date.today() + timedelta(days=1)
    )


# Guest information
st.markdown("### 👤 Guest Information")

adults = st.number_input(
    "Adults (maximum of 10 adults)",
    min_value=1,
    value=2,
    step=1
)

children = st.number_input(
    "Children (maximum of 9 children)",
    min_value=0,
    value=0,
    step=1
)

babies = st.number_input(
    "Babies (maximum of 5 babies)",
    min_value=0,
    value=0,
    step=1
)

country_display = st.selectbox(
    "Country",
    options=sorted(COUNTRY_OPTIONS.keys())
)

is_repeated_guest_label = st.selectbox(
    "Guest status",
    options=[
        "New Guest",
        "Repeated Guest"
    ]
)

is_repeated_guest = (
    1 if is_repeated_guest_label == "Repeated Guest" else 0
)

if is_repeated_guest_label == "Repeated Guest":

    previous_cancellations = st.number_input(
        "Previous cancellations (maximum of 26 previous cancellations)",
        min_value=0,
        value=0,
        step=1
    )
    if previous_cancellations > 26:
        st.error(
            "Number of previous cancellations too high. "
            "Please enter a value of 26 or lower."
    )

    previous_bookings_not_canceled = st.number_input(
        "Previous bookings not cancelled (maximum of 72 previous bookings not cancelled)",
        min_value=0,
        value=1,
        step=1
    )
    if previous_bookings_not_canceled > 72:
        st.error(
            "Number of previous completed bookings too high. "
            "Please enter a value of 72 or lower."
        )
else:
    previous_cancellations = 0
    previous_bookings_not_canceled = 0

# Booking information
st.markdown("### 🛏️ Booking Details")

market_segment_display = st.selectbox(
    "Booking Channel",
    options=[
        "Online Travel Agency (OTA)",
        "Offline Travel Agency / Tour Operator",
        "Groups",
        "Direct Booking",
        "Corporate",
        "Complimentary",
        "Aviation"
    ]
)

market_segment_mapping = {
    "Online Travel Agency (OTA)": "Online TA",
    "Offline Travel Agency / Tour Operator": "Offline TA/TO",
    "Direct Booking": "Direct",
    "Corporate": "Corporate",
    "Groups": "Groups",
    "Complimentary": "Complementary",
    "Aviation": "Aviation",
}

market_segment = market_segment_mapping[market_segment_display]

adr = st.slider(
    "Room Average Daily Rate (ADR)",
    min_value=0,
    max_value=5400,
    value=100,
    step=1
)

required_car_parking_spaces = st.number_input(
    "Required car parking spaces (maximum of 8 parking spaces)",
    min_value=0,
    value=0,
    step=1
)

if required_car_parking_spaces > 8:
    st.error(
        "Number of car parking space too high. "
        "Please enter a value of 8 or lower."
    )

    
total_of_special_requests = st.number_input(
    "Number of special requests (maximum of 5 special requests)",
    min_value=0,
    value=0,
    step=1
)

if total_of_special_requests > 5:
        st.error(
            "Number of special requests too high. "
            "Please enter a value of 5 or lower."
        )

# Agent and company
st.markdown("### 💼 Commercial Information")

has_agent = st.checkbox(
    "Booking made through an agent"
)

has_company = st.checkbox(
    "Booking associated with a company"
)

submit_button = st.button(
    "🔍 Analyze Booking",
    type="primary",
    use_container_width=True
)

if submit_button:

    # -------------------------
    # Input validation
    # -------------------------

    if check_out <= check_in:
        st.error("Check Out date must be after Check In date.")
        st.stop()

    if adults > 10:
        st.error("Number of adults too high. Please choose a maximum of 10 adults.")
        st.stop()

    if children > 9:
        st.error("Number of children too high. Please choose a maximum of 9 children.")
        st.stop()

    if babies > 5:
        st.error("Number of babies too high. Please choose a maximum of 5 babies.")
        st.stop()

    if adr < 0:
        st.error("ADR cannot be negative.")
        st.stop()

    country = COUNTRY_OPTIONS[country_display]

    if previous_cancellations > 26:
        st.error(
            "Number of previous cancellations too high. "
            "Please enter a value of 26 or lower."
        )
        st.stop()

    if previous_bookings_not_canceled > 72:
        st.error(
            "Number of previous completed bookings too high. "
            "Please enter a value of 72 or lower."
        )
        st.stop()

    if required_car_parking_spaces > 8:
        st.error(
            "Number of car parking space too high. "
            "Please enter a value of 8 or lower."
        )
        st.stop()

    if total_of_special_requests > 5:
        st.error(
            "Number of special requests too high. "
            "Please enter a value of 5 or lower."
        )

    # -------------------------
    # Feature engineering
    # -------------------------

    arrival_date_month = check_in.strftime("%B")
    arrival_date_day_of_month = check_in.day

    stays_in_weekend_nights = 0
    stays_in_week_nights = 0

    current_date = check_in

    while current_date < check_out:
        if current_date.weekday() >= 5:
            stays_in_weekend_nights += 1
        else:
            stays_in_week_nights += 1

        current_date += timedelta(days=1)
        
    total_nights = (
        stays_in_weekend_nights
        + stays_in_week_nights
    )

    if total_nights == 0:
        st.warning(
            "The reservation must contain at least one night."
        )
        st.stop()

    # -------------------------
    # Create booking DataFrame
    # -------------------------

    new_booking = pd.DataFrame(
        {
            "hotel": [hotel],
            "lead_time": [lead_time],
            "arrival_date_month": [arrival_date_month],
            "arrival_date_day_of_month": [
                arrival_date_day_of_month
            ],
            "stays_in_weekend_nights": [
                stays_in_weekend_nights
            ],
            "stays_in_week_nights": [
                stays_in_week_nights
            ],
            "adults": [adults],
            "children": [children],
            "babies": [babies],
            "country": [country],
            "market_segment": [market_segment],
            "is_repeated_guest": [
                is_repeated_guest
            ],
            "previous_cancellations": [
                previous_cancellations
            ],
            "previous_bookings_not_canceled": [
                previous_bookings_not_canceled
            ],
            "has_agent": [int(has_agent)],
            "has_company": [int(has_company)],
            "adr": [adr],
            "required_car_parking_spaces": [
                required_car_parking_spaces
            ],
            "total_of_special_requests": [
                total_of_special_requests
            ]
        }
    )

    # Ensure the new data follows the same column order used during model training.
    if hasattr(model, "feature_names_in_"):
        expected_columns = list(
            model.feature_names_in_
        )

        missing_columns = [
            column
            for column in expected_columns
            if column not in new_booking.columns
        ]

        if missing_columns:
            st.error(
                "The application is missing model inputs: "
                + ", ".join(missing_columns)
            )
            st.stop()

        new_booking = new_booking[
            expected_columns
        ]

    try:
        cancellation_probability = (
            model.predict_proba(new_booking)[0, 1]
        )

    except Exception as error:
        st.error(
            f"The prediction could not be completed: {error}"
        )
        st.stop()

    predicted_class = int(
        cancellation_probability
        >= DECISION_THRESHOLD
    )

    probability_percentage = (
        cancellation_probability * 100
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.divider()

    st.subheader("🎯 Prediction")

    col1, col2 = st.columns([1.3, 2])

    with col1:

        st.markdown(
            "<p style='font-size:18px; margin-bottom:5px;'>Estimated Cancellation Probability</p>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <h1 style="
                color:#2F5D62;
                font-size:40px;
                margin-top:0px;
                margin-bottom:0px;
                font-weight:700;
            ">
            {probability_percentage:.1f}%
            </h1>
            """,
            unsafe_allow_html=True
        )

    with col2:

        if cancellation_probability < 0.30:
            st.success(
                "🟢 **Low Risk**\n\n"
                "No immediate action is required"
            )

        elif cancellation_probability < 0.70:
            st.warning(
                "🟡 **Medium Risk**\n\n"
                "A proactive follow-up is recommended"
            )

        else:
            st.error(
                "🔴 **High Risk**\n\n"
                "Immediate follow-up is recommended"
            )
    # -----------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # -----------------------------------------------------

    recommendations = []


    # 1. Booking follow-up
    booking_follow_up_reasons = []


    # Define main recommendation according to cancellation risk
    if cancellation_probability < 0.30:

        recommendation_text = (
            "No immediate follow-up is required."
        )


    elif cancellation_probability < 0.70:

        recommendation_text = (
            "Consider contacting the guest before arrival "
            "to reconfirm the reservation."
        )

        risk_level = "Medium"


    else:

        recommendation_text = (
            "Contact the guest proactively to reconfirm the reservation "
            "and reinforce the booking commitment."
        )

        risk_level = "High"


    # Add indicators and actions for Medium and High risk
    if cancellation_probability >= 0.30:

        if previous_cancellations > 0:
            booking_follow_up_reasons.append(
                "The guest has previous cancellations. "
                "Review guest's previous reservations."
            )

        if lead_time > 180:
            booking_follow_up_reasons.append(
                "The booking was made more than 180 days in advance."
            )

        elif lead_time > 90:
            booking_follow_up_reasons.append(
                "The booking has a long lead time."
            )

        if has_agent:
            booking_follow_up_reasons.append(
                "The booking was made through an agent."
            )

        if market_segment == "Groups":
            booking_follow_up_reasons.append(
                "The booking is part of a group reservation. "
                "Consider confirming the reservation details with the group organizer."
            )

        if total_of_special_requests == 0:
            booking_follow_up_reasons.append(
                "The booking has no special requests."
            )

        # Add key risk indicators
        if booking_follow_up_reasons:

            recommendation_text += (
                "\n\n**Key risk indicators**\n"
            )

            for reason in booking_follow_up_reasons:
                recommendation_text += f"- {reason}\n"

    # Add Booking follow-up card
    if cancellation_probability >= 0.30:

        recommendations.append(
            {
                "title": "Booking follow-up",
                "text": recommendation_text
            }
        )

    # 2. Guest loyalty
    if previous_bookings_not_canceled > 0:
        recommendations.append(
            {
                "title": "Guest loyalty",
                "text": (
                    "This guest has successfully completed previous stays. "
                    "Consider personalized communication to reinforce guest loyalty."
                )
            }
        )


    # Show recommendations
    if recommendations:

        st.subheader("💡 Recommended Actions")

        if recommendations:
            icons = {
                "Booking follow-up": "📞",
                "Guest loyalty": "⭐"
            }

            for recommendation in recommendations:

                with st.container(border=True):

                    icon = icons.get(
                        recommendation["title"],
                        "💡"
                    )

                    st.markdown(
                        f"**{icon} {recommendation['title']}**"
                    )

                    st.markdown(
                        recommendation["text"]
                    )

        else:
            st.success(
                "No specific preventive action was identified "
                "for this booking."
            )
        st.caption(
            "ℹ️ **Disclaimer:** This prediction is intended to support "
            "decision-making and should be interpreted together with "
            "operational knowledge and professional judgment."
    )