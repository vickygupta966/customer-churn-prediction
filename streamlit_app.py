# ============================================================
# CUSTOMER CHURN AI PLATFORM
# Streamlit Frontend
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

BASE_DIR = Path(__file__).resolve().parent

# Try multiple common dataset locations
DATA_PATHS = [
    BASE_DIR / "data" / "customer_churn.csv",
    BASE_DIR / "data" / "customer_churn_cleaned.csv",
    BASE_DIR / "customer_churn.csv",
    BASE_DIR / "customer_churn_cleaned.csv",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-title {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .app-subtitle {
        text-align: center;
        color: #9aa0a6;
        font-size: 18px;
        margin-bottom: 40px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .risk-critical {
        background-color: #472226;
        border-left: 5px solid #ff4b4b;
        padding: 25px;
        border-radius: 10px;
    }

    .risk-high {
        background-color: #493d1f;
        border-left: 5px solid #ffa500;
        padding: 25px;
        border-radius: 10px;
    }

    .risk-medium {
        background-color: #403f22;
        border-left: 5px solid #ffd700;
        padding: 25px;
        border-radius: 10px;
    }

    .risk-low {
        background-color: #193b2b;
        border-left: 5px solid #00d084;
        padding: 25px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_dataset():

    for path in DATA_PATHS:

        if path.exists():
            return path

    return None


@st.cache_data
def load_dataset(path):

    df = pd.read_csv(path)

    return df


def normalize_yes_no(value):
    """Normalize common binary values to the 0/1 format expected by the API."""
    if pd.isna(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if int(value) in (0, 1):
            return int(value)
        raise ValueError("Binary fields must contain 0/1 or Yes/No values.")

    value = str(value).strip().lower()

    if value in ["yes", "y", "true", "1"]:
        return 1

    if value in ["no", "n", "false", "0"]:
        return 0

    raise ValueError(
        f"Invalid binary value '{value}'. Use Yes/No or 1/0."
    )


def normalize_churn_column(df):

    df = df.copy()

    if "churn" not in df.columns:
        return df

    if df["churn"].dtype == "object":

        df["churn"] = (
            df["churn"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "yes": 1,
                    "no": 0,
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "churn": 1,
                    "churned": 1,
                    "stay": 0,
                    "stayed": 0,
                }
            )
        )

    df["churn"] = pd.to_numeric(
        df["churn"],
        errors="coerce",
    )

    return df


def get_api_health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except Exception:
        return None


def show_api_status():

    health = get_api_health()

    if health:

        if health.get("model_loaded"):

            st.sidebar.success(
                "🟢 API Connected"
            )

            api_threshold = health.get("threshold")
            if api_threshold is not None:
                st.sidebar.caption(
                    f"Classification threshold: {float(api_threshold):.2f}"
                )

        else:

            st.sidebar.warning(
                "🟡 API Running - Model Not Loaded"
            )

    else:

        st.sidebar.error(
            "🔴 API Offline"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 📊 Customer Churn AI"
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🔮 Prediction",
            "📊 Business Analytics",
            "📈 Model Performance",
            "📁 Batch Prediction",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div style="
        background-color:#29486d;
        padding:20px;
        border-radius:10px;
        ">

        <h4>ML Pipeline</h4>

        Feature Engineering<br>
        ↓<br>
        Preprocessing<br>
        ↓<br>
        Tuned ML Model<br>
        ↓<br>
        Threshold Optimization<br>
        ↓<br>
        Churn Prediction

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    show_api_status()


# ============================================================
# COMMON HEADER
# ============================================================

def show_header():

    st.markdown(
        '<div class="app-title">📊 Customer Churn AI Platform</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">'
        "Predict customer churn probability and generate actionable retention strategies"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

def show_prediction():

    show_header()

    st.markdown(
        '<div class="section-title">👤 Customer Information</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Customer information
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        customer_id = st.text_input(
            "Customer ID",
            value="TEST_001",
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=34,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
            ],
        )

        senior_citizen_ui = st.selectbox(
            "Senior Citizen",
            [
                "No",
                "Yes",
            ],
        )

    with col2:

        has_partner_ui = st.selectbox(
            "Has Partner",
            [
                "No",
                "Yes",
            ],
        )

        has_dependents_ui = st.selectbox(
            "Has Dependents",
            [
                "No",
                "Yes",
            ],
        )

        tenure_months = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=120,
            value=3,
            step=1,
        )

        contract_type = st.selectbox(
            "Contract Type",
            [
                "Month-to-Month",
                "One Year",
                "Two Year",
            ],
        )

    with col3:

        phone_service_ui = st.selectbox(
            "Phone Service",
            [
                "No",
                "Yes",
            ],
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service",
            ],
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber Optic",
                "No",
            ],
        )

    # --------------------------------------------------------
    # Services
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🌐 Services</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

    with col2:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service",
            ],
        )

    with col3:

        paperless_billing_ui = st.selectbox(
            "Paperless Billing",
            [
                "No",
                "Yes",
            ],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic Check",
                "Mailed Check",
                "Bank Transfer",
                "Credit Card",
            ],
        )

    # --------------------------------------------------------
    # Billing & Usage
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">💰 Billing & Usage</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=95.70,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=286.50,
            step=1.0,
        )

    with col2:

        data_usage_gb = st.number_input(
            "Data Usage (GB)",
            min_value=0.0,
            value=42.30,
            step=1.0,
        )

        num_tech_tickets = st.number_input(
            "Technical Support Tickets",
            min_value=0,
            value=4,
            step=1,
        )

    with col3:

        num_admin_tickets = st.number_input(
            "Administrative Tickets",
            min_value=0,
            value=2,
            step=1,
        )

        satisfaction_score = st.slider(
            "Satisfaction Score",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
        )

    st.divider()

    # --------------------------------------------------------
    # Convert UI Yes/No to API integers
    # --------------------------------------------------------

    senior_citizen = normalize_yes_no(
        senior_citizen_ui
    )

    has_partner = normalize_yes_no(
        has_partner_ui
    )

    has_dependents = normalize_yes_no(
        has_dependents_ui
    )

    phone_service = normalize_yes_no(
        phone_service_ui
    )

    paperless_billing = normalize_yes_no(
        paperless_billing_ui
    )

    # --------------------------------------------------------
    # API Payload
    # --------------------------------------------------------

    payload = {
        "customer_id": customer_id,
        "age": int(age),
        "gender": gender,

        "senior_citizen": senior_citizen,
        "has_partner": has_partner,
        "has_dependents": has_dependents,

        "tenure_months": int(tenure_months),
        "contract_type": contract_type,

        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,

        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,

        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,

        "paperless_billing": paperless_billing,
        "payment_method": payment_method,

        "monthly_charges": float(
            monthly_charges
        ),

        "total_charges": float(
            total_charges
        ),

        "data_usage_gb": float(
            data_usage_gb
        ),

        "num_tech_tickets": int(
            num_tech_tickets
        ),

        "num_admin_tickets": int(
            num_admin_tickets
        ),

        "satisfaction_score": float(
            satisfaction_score
        ),
    }

    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True,
        type="primary",
    ):

        with st.spinner(
            "Running churn prediction..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

                # ------------------------------------------------
                # Success
                # ------------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "✅ Prediction completed successfully!"
                    )

                    st.divider()

                    st.markdown(
                        '<div class="section-title">'
                        "📈 Prediction Result"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    raw_probability = result.get(
                        "churn_probability_percent"
                    )

                    if raw_probability is None:
                        raw_probability = result.get(
                            "churn_probability",
                            0,
                        )
                        # API currently returns a percentage, but this also
                        # supports a probability in the 0-1 range.
                        raw_probability = (
                            float(raw_probability) * 100
                            if 0 <= float(raw_probability) <= 1
                            else float(raw_probability)
                        )

                    probability = float(raw_probability)

                    prediction = result.get(
                        "prediction",
                        "Unknown",
                    )

                    risk = result.get(
                        "risk_category",
                        "Unknown",
                    )

                    threshold = result.get(
                        "classification_threshold",
                        0.50,
                    )

                    # --------------------------------------------
                    # Result metrics
                    # --------------------------------------------

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Churn Probability",
                            f"{probability:.2f}%",
                        )

                    with col2:

                        st.metric(
                            "Prediction",
                            prediction,
                        )

                    with col3:

                        st.metric(
                            "Risk Category",
                            risk,
                        )

                    with col4:

                        st.metric(
                            "Threshold",
                            f"{float(threshold):.2f}",
                        )

                    # --------------------------------------------
                    # Risk Assessment
                    # --------------------------------------------

                    st.markdown(
                        "### 🎯 Risk Assessment"
                    )

                    risk_class = (
                        str(risk)
                        .lower()
                        .replace(" ", "-")
                    )

                    st.markdown(
                        f"""
                        <div class="risk-{risk_class}">
                            <h2>{risk.upper()} RISK</h2>
                            <p>
                            Customer has a
                            <strong>{probability:.2f}%</strong>
                            probability of churning.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # --------------------------------------------
                    # Recommended Actions
                    # --------------------------------------------

                    st.markdown(
                        "### 🎯 Recommended Business Action"
                    )

                    actions = result.get(
                        "recommended_actions"
                    )

                    if not actions:
                        single_action = result.get(
                            "recommended_action"
                        )
                        actions = [single_action] if single_action else []

                    if isinstance(actions, str):
                        actions = [actions]

                    if actions:
                        for action in actions:
                            st.info(f"• {action}")
                    else:
                        st.info(
                            "No recommended actions returned by API."
                        )

                    # --------------------------------------------
                    # API response
                    # --------------------------------------------

                    with st.expander(
                        "View API Response"
                    ):

                        st.json(result)

                # ------------------------------------------------
                # Validation error
                # ------------------------------------------------

                elif response.status_code == 422:

                    st.error(
                        "❌ API Validation Error (422)"
                    )

                    try:

                        error_data = response.json()

                        st.json(
                            error_data
                        )

                    except Exception:

                        st.code(
                            response.text
                        )

                    st.warning(
                        "The frontend payload does not match the "
                        "FastAPI CustomerData schema."
                    )

                # ------------------------------------------------
                # Other API errors
                # ------------------------------------------------

                else:

                    st.error(
                        f"❌ API Error: {response.status_code}"
                    )

                    try:

                        st.json(
                            response.json()
                        )

                    except Exception:

                        st.code(
                            response.text
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "🔴 Cannot connect to FastAPI."
                )

                st.info(
                    "Make sure FastAPI is running on "
                    f"{API_URL}"
                )

                st.code(
                    "uvicorn api:app --reload --port 8000"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ API request timed out."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

def show_business_analytics():

    show_header()

    st.markdown(
        '<div class="section-title">📊 Business Analytics</div>',
        unsafe_allow_html=True,
    )

    dataset_path = find_dataset()

    if dataset_path is None:

        st.error(
            "Dataset not found."
        )

        st.info(
            "Place your CSV inside the data folder."
        )

        st.code(
            "data/customer_churn.csv"
        )

        return

    try:

        df = load_dataset(
            str(dataset_path)
        )

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )

        return

    df = normalize_churn_column(df)

    if "churn" not in df.columns:

        st.error(
            "Your dataset must contain a 'churn' column "
            "for Business Analytics."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(df.columns)
        )

        return

    df = df.dropna(
        subset=["churn"]
    )

    df["churn"] = df[
        "churn"
    ].astype(int)

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "monthly_charges",
        "total_charges",
        "tenure_months",
        "satisfaction_score",
        "num_tech_tickets",
        "num_admin_tickets",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_customers = len(df)

    churned_customers = int(
        df["churn"].sum()
    )

    stayed_customers = (
        total_customers
        - churned_customers
    )

    churn_rate = (
        churned_customers
        / total_customers
        * 100
        if total_customers
        else 0
    )

    if "monthly_charges" in df.columns:

        avg_monthly_charges = (
            df["monthly_charges"]
            .mean()
        )

    else:

        avg_monthly_charges = 0

    # --------------------------------------------------------
    # High-risk business segment
    # --------------------------------------------------------

    high_risk_customers = 0

    if (
        "satisfaction_score" in df.columns
    ):

        high_risk_customers = len(
            df[
                (
                    df[
                        "satisfaction_score"
                    ] < 3
                )
                & (
                    df["churn"] == 1
                )
            ]
        )

    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    st.markdown(
        "### 📌 Business KPIs"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}",
        )

    with col2:

        st.metric(
            "Churned Customers",
            f"{churned_customers:,}",
        )

    with col3:

        st.metric(
            "Stayed Customers",
            f"{stayed_customers:,}",
        )

    with col4:

        st.metric(
            "Overall Churn Rate",
            f"{churn_rate:.2f}%",
        )

    with col5:

        st.metric(
            "Avg Monthly Charges",
            f"${avg_monthly_charges:.2f}",
        )

    st.divider()

    # --------------------------------------------------------
    # Churn Distribution
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Churn Distribution"
    )

    churn_distribution = pd.Series(
        {
            "Stayed": stayed_customers,
            "Churned": churned_customers,
        }
    )

    st.bar_chart(
        churn_distribution
    )

    # --------------------------------------------------------
    # Contract Analysis
    # --------------------------------------------------------

    if "contract_type" in df.columns:

        st.markdown(
            "### 📄 Churn by Contract Type"
        )

        contract = (
            df.groupby(
                "contract_type"
            )["churn"]
            .agg(
                customers="count",
                churned="sum",
            )
        )

        contract["churn_rate"] = (
            contract["churned"]
            / contract["customers"]
            * 100
        )

        st.bar_chart(
            contract[
                "churn_rate"
            ]
        )

        display_contract = contract.copy()

        display_contract[
            "churn_rate"
        ] = display_contract[
            "churn_rate"
        ].round(2)

        st.dataframe(
            display_contract,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Internet Service
    # --------------------------------------------------------

    if "internet_service" in df.columns:

        st.markdown(
            "### 🌐 Churn by Internet Service"
        )

        internet = (
            df.groupby(
                "internet_service"
            )["churn"]
            .agg(
                customers="count",
                churned="sum",
            )
        )

        internet["churn_rate"] = (
            internet["churned"]
            / internet["customers"]
            * 100
        )

        st.bar_chart(
            internet[
                "churn_rate"
            ]
        )

    # --------------------------------------------------------
    # Tenure Analysis
    # --------------------------------------------------------

    if "tenure_months" in df.columns:

        st.markdown(
            "### ⏳ Churn by Customer Tenure"
        )

        df["tenure_group"] = pd.cut(
            df["tenure_months"],
            bins=[
                -np.inf,
                12,
                24,
                48,
                np.inf,
            ],
            labels=[
                "0-12 months",
                "13-24 months",
                "25-48 months",
                "49+ months",
            ],
        )

        tenure = (
            df.groupby(
                "tenure_group",
                observed=False,
            )["churn"]
            .agg(
                customers="count",
                churned="sum",
            )
        )

        tenure["churn_rate"] = (
            tenure["churned"]
            / tenure["customers"]
            * 100
        )

        st.bar_chart(
            tenure[
                "churn_rate"
            ]
        )

    # --------------------------------------------------------
    # Payment Method
    # --------------------------------------------------------

    if "payment_method" in df.columns:

        st.markdown(
            "### 💳 Churn by Payment Method"
        )

        payment = (
            df.groupby(
                "payment_method"
            )["churn"]
            .agg(
                customers="count",
                churned="sum",
            )
        )

        payment["churn_rate"] = (
            payment["churned"]
            / payment["customers"]
            * 100
        )

        st.bar_chart(
            payment[
                "churn_rate"
            ]
        )

    # --------------------------------------------------------
    # Satisfaction
    # --------------------------------------------------------

    if "satisfaction_score" in df.columns:

        st.markdown(
            "### 😊 Churn by Satisfaction Score"
        )

        satisfaction = (
            df.groupby(
                "satisfaction_score"
            )["churn"]
            .agg(
                customers="count",
                churned="sum",
            )
        )

        satisfaction["churn_rate"] = (
            satisfaction["churned"]
            / satisfaction["customers"]
            * 100
        )

        st.line_chart(
            satisfaction[
                "churn_rate"
            ]
        )

    # --------------------------------------------------------
    # Monthly Charges
    # --------------------------------------------------------

    if "monthly_charges" in df.columns:

        st.markdown(
            "### 💰 Average Monthly Charges by Churn"
        )

        charge_analysis = (
            df.groupby(
                "churn"
            )[
                "monthly_charges"
            ]
            .mean()
        )

        charge_analysis.index = [
            "Stayed"
            if value == 0
            else "Churned"
            for value in charge_analysis.index
        ]

        st.bar_chart(
            charge_analysis
        )

    # --------------------------------------------------------
    # Support Tickets
    # --------------------------------------------------------

    if (
        "num_tech_tickets" in df.columns
        and
        "num_admin_tickets" in df.columns
    ):

        st.markdown(
            "### 🎫 Churn by Support Tickets"
        )

        df["total_tickets"] = (
            df["num_tech_tickets"].fillna(0)
            +
            df["num_admin_tickets"].fillna(0)
        )

        ticket_analysis = (
            df.groupby(
                "total_tickets"
            )["churn"]
            .mean()
            * 100
        )

        ticket_analysis = (
            ticket_analysis
            .to_frame(
                "Churn Rate (%)"
            )
        )

        st.line_chart(
            ticket_analysis
        )

    # --------------------------------------------------------
    # Business Insights
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 💡 Business Insights"
    )

    insights = []

    if "contract_type" in df.columns:

        contract_rate = (
            df.groupby(
                "contract_type"
            )["churn"]
            .mean()
            * 100
        )

        if not contract_rate.empty:

            highest_contract = (
                contract_rate.idxmax()
            )

            highest_contract_rate = (
                contract_rate.max()
            )

            insights.append(
                f"📄 **Contract:** "
                f"{highest_contract} has the highest "
                f"churn rate "
                f"({highest_contract_rate:.2f}%)."
            )

    if "internet_service" in df.columns:

        internet_rate = (
            df.groupby(
                "internet_service"
            )["churn"]
            .mean()
            * 100
        )

        if not internet_rate.empty:

            highest_internet = (
                internet_rate.idxmax()
            )

            highest_internet_rate = (
                internet_rate.max()
            )

            insights.append(
                f"🌐 **Internet Service:** "
                f"{highest_internet} has the highest "
                f"churn rate "
                f"({highest_internet_rate:.2f}%)."
            )

    if (
        "satisfaction_score" in df.columns
    ):

        low_satisfaction = df[
            df[
                "satisfaction_score"
            ] < 3
        ]

        if len(low_satisfaction) > 0:

            low_sat_churn = (
                low_satisfaction[
                    "churn"
                ].mean()
                * 100
            )

            insights.append(
                f"😊 **Satisfaction:** "
                f"Customers with satisfaction below 3 "
                f"have a churn rate of "
                f"{low_sat_churn:.2f}%."
            )

    if (
        "tenure_months" in df.columns
    ):

        new_customers = df[
            df["tenure_months"] <= 12
        ]

        if len(new_customers) > 0:

            new_customer_churn = (
                new_customers[
                    "churn"
                ].mean()
                * 100
            )

            insights.append(
                f"⏳ **Tenure:** "
                f"Customers with 12 months or less "
                f"tenure have a churn rate of "
                f"{new_customer_churn:.2f}%."
            )

    if high_risk_customers > 0:

        insights.append(
            f"🚨 **Retention Opportunity:** "
            f"{high_risk_customers:,} customers have "
            f"low satisfaction and have churned."
        )

    for insight in insights:

        st.info(
            insight
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def show_model_performance():

    show_header()

    st.markdown(
        '<div class="section-title">📈 Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Model Comparison"
    )

    # Model comparison metrics from the untouched test evaluation.
    # Model selection was based only on 5-fold CV ROC-AUC.
    model_data = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Neural Network",
                "Random Forest",
                "XGBoost",
            ],
            "CV ROC-AUC Mean": [
                0.8075,
                0.7981,
                0.7960,
                0.7912,
            ],
            "CV ROC-AUC Std": [
                0.0049,
                0.0036,
                0.0044,
                0.0063,
            ],
            "Accuracy (0.50)": [
                0.7232,
                0.7700,
                0.7475,
                0.7190,
            ],
            "Precision (0.50)": [
                0.5009,
                0.6193,
                0.5374,
                0.4956,
            ],
            "Recall (0.50)": [
                0.7604,
                0.4441,
                0.6477,
                0.7054,
            ],
            "F1 (0.50)": [
                0.6039,
                0.5173,
                0.5874,
                0.5822,
            ],
            "ROC-AUC": [
                0.8100,
                0.8013,
                0.7957,
                0.7933,
            ],
            "Train Time (s)": [
                9.24,
                19.11,
                21.87,
                9.28,
            ],
        }
    )

    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = (
        model_data
        .sort_values(
            "CV ROC-AUC Mean",
            ascending=False,
        )
        .iloc[0]
    )

    st.success(
        "🏆 Best model based on 5-fold CV ROC-AUC: "
        f"**{best_model['Model']}**"
    )

    st.caption(
        "The final test set is used only for final evaluation, not model selection."
    )

    # --------------------------------------------------------
    # Final production evaluation
    # --------------------------------------------------------

    st.markdown("### 🎯 Final Production Evaluation")

    st.info(
        "The production decision threshold was optimized to 0.54 using "
        "5-fold out-of-fold predictions on the training data only. "
        "The final test set remained untouched until this evaluation."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("CV ROC-AUC", "0.8075")

    with col2:
        st.metric("Test ROC-AUC", "0.8100")

    with col3:
        st.metric("Production F1", "0.6080")

    with col4:
        st.metric("Threshold", "0.54")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "0.7438")

    with col2:
        st.metric("Precision", "0.5282")

    with col3:
        st.metric("Recall", "0.7162")

    st.markdown("### 🔢 Confusion Matrix")

    confusion_df = pd.DataFrame(
        {
            "Predicted Stay": [2180, 315],
            "Predicted Churn": [710, 795],
        },
        index=["Actual Stay", "Actual Churn"],
    )

    st.dataframe(
        confusion_df,
        use_container_width=True,
    )

    st.caption(
        "TN = 2,180 | FP = 710 | FN = 315 | TP = 795"
    )

    st.markdown("### 📊 Threshold Comparison")

    threshold_df = pd.DataFrame(
        {
            "Metric": ["F1", "Precision", "Recall"],
            "Default 0.50": [0.6037, 0.5014, 0.7452],
            "Optimized 0.54": [0.6080, 0.5282, 0.7162],
        }
    ).set_index("Metric")

    st.dataframe(
        threshold_df,
        use_container_width=True,
    )

    st.success(
        "✅ The optimized threshold improves F1 from 0.6037 to 0.6080 "
        "while keeping recall at 0.7162."
    )

    st.divider()

    # --------------------------------------------------------
    # ROC-AUC chart
    # --------------------------------------------------------

    st.markdown("### ROC-AUC Comparison")

    roc_chart = model_data[
        [
            "Model",
            "ROC-AUC",
        ]
    ].set_index("Model")

    st.bar_chart(roc_chart)

    st.markdown(
        """
        **Interpretation**

        ROC-AUC measures how well the model separates
        customers who churn from customers who stay.

        A higher value generally indicates better
        discrimination ability.

        **Model selection:** Logistic Regression was selected using
        5-fold cross-validation ROC-AUC (0.8075).

        **Threshold optimization:** 0.54 was selected using
        5-fold out-of-fold training predictions and F1.

        **Final evaluation:** The 4,000-row test set was used only once
        for final performance reporting.
        """
    )


# ============================================================
# BATCH PREDICTION
# ============================================================

def show_batch_prediction():

    show_header()

    st.markdown(
        '<div class="section-title">📁 Batch Prediction</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a CSV containing customer information "
        "to generate churn predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin."
        )

        return

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read CSV: {e}"
        )

        return

    st.success(
        f"Loaded {len(df):,} records."
    )

    st.markdown(
        "### Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )

    required_columns = [
        "age",
        "gender",
        "senior_citizen",
        "has_partner",
        "has_dependents",
        "tenure_months",
        "contract_type",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "data_usage_gb",
        "num_tech_tickets",
        "num_admin_tickets",
        "satisfaction_score",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        st.error(
            "Missing required columns:"
        )

        st.write(
            missing
        )

        return

    if st.button(
        "🚀 Run Batch Prediction",
        type="primary",
        use_container_width=True,
    ):

        results = []

        progress = st.progress(
            0
        )

        status = st.empty()

        total_rows = len(df)

        for index, row in df.iterrows():

            try:

                customer_id = row.get(
                    "customer_id",
                    f"BATCH_{index + 1:05d}",
                )

                payload = {
                    "customer_id": str(
                        customer_id
                    ),

                    "age": int(
                        row["age"]
                    ),

                    "gender": str(
                        row["gender"]
                    ),

                    "senior_citizen":
                        normalize_yes_no(
                            row["senior_citizen"]
                        ),

                    "has_partner":
                        normalize_yes_no(
                            row["has_partner"]
                        ),

                    "has_dependents":
                        normalize_yes_no(
                            row["has_dependents"]
                        ),

                    "tenure_months": int(
                        row["tenure_months"]
                    ),

                    "contract_type": str(
                        row["contract_type"]
                    ),

                    "phone_service":
                        normalize_yes_no(
                            row["phone_service"]
                        ),

                    "multiple_lines": str(
                        row["multiple_lines"]
                    ),

                    "internet_service": str(
                        row["internet_service"]
                    ),

                    "online_security": str(
                        row["online_security"]
                    ),

                    "online_backup": str(
                        row["online_backup"]
                    ),

                    "device_protection": str(
                        row["device_protection"]
                    ),

                    "tech_support": str(
                        row["tech_support"]
                    ),

                    "streaming_tv": str(
                        row["streaming_tv"]
                    ),

                    "streaming_movies": str(
                        row["streaming_movies"]
                    ),

                    "paperless_billing":
                        normalize_yes_no(
                            row["paperless_billing"]
                        ),

                    "payment_method": str(
                        row["payment_method"]
                    ),

                    "monthly_charges": float(
                        row["monthly_charges"]
                    ),

                    "total_charges": float(
                        row["total_charges"]
                    ),

                    "data_usage_gb": float(
                        row["data_usage_gb"]
                    ),

                    "num_tech_tickets": int(
                        row["num_tech_tickets"]
                    ),

                    "num_admin_tickets": int(
                        row["num_admin_tickets"]
                    ),

                    "satisfaction_score": float(
                        row["satisfaction_score"]
                    ),
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:

                    result = response.json()

                    results.append(
                        {
                            "customer_id":
                                result.get(
                                    "customer_id"
                                ),

                            "churn_probability":
                                result.get(
                                    "churn_probability_percent"
                                ),

                            "prediction":
                                result.get(
                                    "prediction"
                                ),

                            "risk_category":
                                result.get(
                                    "risk_category"
                                ),

                            "threshold":
                                result.get(
                                    "classification_threshold"
                                ),
                        }
                    )

                else:

                    results.append(
                        {
                            "customer_id":
                                customer_id,

                            "churn_probability":
                                None,

                            "prediction":
                                "API Error",

                            "risk_category":
                                None,

                            "threshold":
                                None,
                        }
                    )

            except Exception as e:

                results.append(
                    {
                        "customer_id":
                            row.get(
                                "customer_id",
                                f"BATCH_{index + 1:05d}",
                            ),

                        "churn_probability":
                            None,

                        "prediction":
                            f"Error: {str(e)}",

                        "risk_category":
                            None,

                        "threshold":
                            None,
                    }
                )

            progress.progress(
                (index + 1)
                / total_rows
            )

            status.text(
                f"Processing "
                f"{index + 1:,} / "
                f"{total_rows:,}"
            )

        result_df = pd.DataFrame(
            results
        )

        st.success(
            "✅ Batch prediction completed!"
        )

        st.markdown(
            "### Prediction Results"
        )

        st.dataframe(
            result_df,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        valid_results = result_df[
            result_df["prediction"].isin(
                [
                    "Will Churn",
                    "Will Stay",
                ]
            )
        ]

        if not valid_results.empty:

            total_predictions = len(
                valid_results
            )

            churn_predictions = len(
                valid_results[
                    valid_results[
                        "prediction"
                    ] == "Will Churn"
                ]
            )

            batch_churn_rate = (
                churn_predictions
                / total_predictions
                * 100
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predictions",
                    f"{total_predictions:,}",
                )

            with col2:

                st.metric(
                    "Predicted Churn",
                    f"{churn_predictions:,}",
                )

            with col3:

                st.metric(
                    "Predicted Churn Rate",
                    f"{batch_churn_rate:.2f}%",
                )

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction Results",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "🔮 Prediction":

    show_prediction()

elif page == "📊 Business Analytics":

    show_business_analytics()

elif page == "📈 Model Performance":

    show_model_performance()

elif page == "📁 Batch Prediction":

    show_batch_prediction()