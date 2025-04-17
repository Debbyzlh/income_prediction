import streamlit as st
import pandas as pd
import numpy as np

log_reg_coefs = {
    "Intercept": -5.13,
    "age": 0.0238,
    "hours.per.week": 0.0290,
    "workclass_X.Local.gov": -0.388,
    "workclass_X.Private": -0.226,
    "workclass_X.Self.emp.not.inc": -0.762,
    "education_X.Bachelors": 0.749,
    "education_X.HS.grad": -0.273,
    "education_X.Masters": 1.02,
    "education_X.Some.college": 0.0362,
    "marital.status_X.Married.civ.spouse": 2.26,
    "marital.status_X.Never.married": -0.432,
    "occupation_X.Craft.repair": 0.0152,
    "occupation_X.Exec.managerial": 0.983,
    "occupation_X.Machine.op.inspsct": -0.422,
    "occupation_X.Other.service": -1.01,
    "occupation_X.Prof.specialty": 1.01,
    "occupation_X.Sales": 0.432,
    "occupation_X.Transport.moving": -0.267,
    "relationship_X.Not.in.family": 0.428,
    "relationship_X.Own.child": -0.990,
    "relationship_X.Unmarried": 0.00249,
    "race_X.Black": 0.00963,
    "race_X.White": 0.137,
    "sex_X.Male": 0.179
}

# Title
st.title("Income Prediction App")

# User Inputs
age = st.slider("Age", 18, 100, 30)
marital_status = st.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent"
])
sex = st.radio("Sex", ["Male", "Female"])
education_num = st.slider("Years of Education", 1, 16, 10)
hours_per_week = st.slider("Hours Worked per Week", 1, 100, 40)

def preprocess_input(age, marital_status, sex, education_num, hours_per_week):
    features = {
        "Intercept": 1,
        "age": age,
        "hours.per.week": hours_per_week,
        "sex_X.Male": 1 if sex == "Male" else 0,
        "marital.status_X.Married.civ.spouse": 1 if marital_status == "Married-civ-spouse" else 0,
        "marital.status_X.Never.married": 1 if marital_status == "Never-married" else 0,
        "workclass_X.Local.gov": 0,
        "workclass_X.Private": 0,
        "workclass_X.Self.emp.not.inc": 0,
        "education_X.Bachelors": 0,
        "education_X.HS.grad": 0,
        "education_X.Masters": 0,
        "education_X.Some.college": 0,
        "occupation_X.Craft.repair": 0,
        "occupation_X.Exec.managerial": 0,
        "occupation_X.Machine.op.inspsct": 0,
        "occupation_X.Other.service": 0,
        "occupation_X.Prof.specialty": 0,
        "occupation_X.Sales": 0,
        "occupation_X.Transport.moving": 0,
        "relationship_X.Not.in.family": 0,
        "relationship_X.Own.child": 0,
        "relationship_X.Unmarried": 0,
        "race_X.Black": 0,
        "race_X.White": 0
    }
    return pd.DataFrame([features])

def predict_income(df, coefs):
    logit = sum(df[col].values[0] * coef for col, coef in coefs.items())
    prob = 1 / (1 + np.exp(-logit))
    return prob

# Prediction
if st.button("Predict Income Level"):
    input_df = preprocess_input(age, marital_status, sex, education_num, hours_per_week)
    probability = predict_income(input_df, log_reg_coefs)
    predicted_class = ">50K" if probability > 0.5 else "<=50K"
    st.success(f"Predicted Income: {predicted_class}")
    st.write("Raw model output (probability):", probability)
    st.write("Log-odds (linear output):", np.log(probability / (1 - probability)))
    st.write("Odds of >50K income:", round(probability / (1 - probability), 2))