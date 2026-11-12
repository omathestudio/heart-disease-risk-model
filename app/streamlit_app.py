"""Heart Disease Risk Model: interactive app.

Enter your details and see the model's estimated risk, plus how the estimate
changes without each risk factor.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# The saved model uses code from src/ (the preprocessing steps), so Python
# needs to be able to find that folder before the model is loaded.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import ORDINAL_FEATURES, THRESHOLD, all_features  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "heart_risk_model.joblib"
AVERAGE_RISK = 0.09          # heart disease rate in the survey sample
NOT_SAID = "Prefer not to say"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, person):
    """Risk (0 to 1) for one person, given as a dict of feature values."""
    row = pd.DataFrame([person], columns=all_features())
    return model.predict_proba(row)[0, 1]


def yes_no(answer):
    return {"Yes": 1, "No": 0}.get(answer, np.nan)


# --- Page setup ---------------------------------------------------------------

st.set_page_config(page_title="Heart Disease Risk Model", page_icon="❤️",
                   layout="centered")

st.title("Heart Disease Risk Model")
st.write(
    "Estimates the chance that someone like you has heart disease, using a "
    "logistic regression model trained on the CDC's 2025 survey of 352,145 "
    "US adults."
)
st.warning(
    "**This is a portfolio project, not a medical tool.** It's based on "
    "self-reported US survey data and shows associations, not causes. "
    "If you're worried about your heart health, speak to a GP or doctor.",
    icon="⚠️",
)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python src/evaluate_final.py` first.")
    st.stop()

model = load_model()


# --- Inputs -------------------------------------------------------------------

st.header("About you")

col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("Age", ORDINAL_FEATURES["age_group"], index=8)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
with col2:
    height_cm = st.number_input("Height (cm)", 120, 220, 168)
    weight_kg = st.number_input("Weight (kg)", 30, 250, 75)

bmi = weight_kg / (height_cm / 100) ** 2
st.caption(f"Your BMI: **{bmi:.1f}**")

st.header("Health")

col1, col2 = st.columns(2)
with col1:
    high_bp = st.radio("Ever told you have high blood pressure?",
                       ["No", "Yes"], horizontal=True)
    diabetes = st.radio("Ever told you have diabetes?", ["No", "Yes"],
                        horizontal=True,
                        help="Diagnosed diabetes only. Pre-diabetes and "
                             "diabetes only during pregnancy count as no.")
with col2:
    cholesterol = st.radio("Ever told you have high cholesterol?",
                           ["No", "Yes", "Never had it checked"],
                           horizontal=True)

st.header("Lifestyle")

col1, col2 = st.columns(2)
with col1:
    smoking = st.selectbox("Smoking", ["Never", "Former", "Current (some days)",
                                       "Current (every day)"])
    exercise = st.radio("Any exercise in the past 30 days (apart from your job)?",
                        ["Yes", "No"], horizontal=True)
with col2:
    heavy_drinker = st.radio(
        "Heavy drinker?", ["No", "Yes"], horizontal=True,
        help="Survey definition: more than 14 drinks a week for men, "
             "more than 7 for women.")

with st.expander("Education and income (optional)"):
    education = st.selectbox("Education",
                             [NOT_SAID] + ORDINAL_FEATURES["education"])
    income = st.selectbox("Household income (US dollars, as in the survey)",
                          [NOT_SAID] + ORDINAL_FEATURES["income"])

person = {
    "bmi": bmi,
    "high_bp": yes_no(high_bp),
    "high_cholesterol": (np.nan if cholesterol == "Never had it checked"
                         else yes_no(cholesterol)),
    "diabetes": yes_no(diabetes),
    "inactive": 1 if exercise == "No" else 0,
    "heavy_drinker": yes_no(heavy_drinker),
    "age_group": age_group,
    "education": None if education == NOT_SAID else education,
    "income": None if income == NOT_SAID else income,
    "sex": sex,
    "smoking_status": smoking,
}


# --- Result -------------------------------------------------------------------

risk = predict(model, person)

st.header("Your estimated risk")

col1, col2, col3 = st.columns(3)
col1.metric("Estimated risk", f"{risk:.1%}")
col2.metric("Survey average", f"{AVERAGE_RISK:.0%}")
col3.metric("Compared with average", f"{risk / AVERAGE_RISK:.1f}×")

st.progress(min(risk, 1.0))

if risk >= THRESHOLD:
    st.info(
        f"This is **above the model's {THRESHOLD:.0%} cut-off**. In the test "
        "data, about 1 in 5 people above this level had heart disease."
    )
else:
    st.success(f"This is **below the model's {THRESHOLD:.0%} cut-off**.")

st.caption(
    "Out of 100 people with these answers, the model estimates about "
    f"**{risk * 100:.0f}** would have heart disease."
)


# --- What's driving it ----------------------------------------------------------

st.header("What's driving this estimate?")
st.write(
    "How the estimate changes for a similar person **without** each risk "
    "factor, with everything else kept the same."
)

what_ifs = []
if person["high_bp"] == 1:
    what_ifs.append(("Normal blood pressure", {"high_bp": 0}))
if person["high_cholesterol"] == 1:
    what_ifs.append(("Normal cholesterol", {"high_cholesterol": 0}))
if person["diabetes"] == 1:
    what_ifs.append(("No diabetes", {"diabetes": 0}))
if smoking.startswith("Current"):
    what_ifs.append(("Former smoker (quit)", {"smoking_status": "Former"}))
if person["inactive"] == 1:
    what_ifs.append(("Some exercise in the past 30 days", {"inactive": 0}))
if bmi > 25:
    what_ifs.append(("BMI of 25", {"bmi": 25.0}))

if what_ifs:
    rows = []
    for label, change in what_ifs:
        new_risk = predict(model, {**person, **change})
        rows.append({"Without this factor": label,
                     "Estimated risk": f"{new_risk:.1%}",
                     "Change": f"{(new_risk - risk) * 100:+.1f} points"})
    st.table(pd.DataFrame(rows).set_index("Without this factor"))
    st.caption(
        "These compare similar people in the survey. They are not predictions "
        "of what would happen if you changed something, because the model "
        "shows associations, not causes."
    )
else:
    st.write("None of the changeable risk factors in the model apply to you. "
             "Your estimate is driven mainly by age and sex.")


# --- About ----------------------------------------------------------------------

with st.expander("About the model"):
    st.markdown(
        """
- **Model:** logistic regression, chosen over gradient boosting because it
  performed the same (ROC-AUC 0.81) and is easier to explain.
- **Test results** on 52,822 people the model never saw: ROC-AUC 0.809,
  catches 75% of heart disease cases at the 10% cut-off, and predicted risks
  within 0.1 percentage points of reality.
- **Heart disease** means ever being told you had coronary heart disease or a
  heart attack. The model estimates whether someone *already has* it, not their
  future risk.
- **Limitations:** self-reported US data; the 10% cut-off flags far fewer cases
  among women and under-45s; no data on ethnicity.

Built by **Nneoma (Oma)**. Code and full write-up:
[github.com/omathestudio/heart-disease-risk-model](https://github.com/omathestudio/heart-disease-risk-model)
        """
    )
