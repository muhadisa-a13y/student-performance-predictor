import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Robust loader: inspects .pkl files and finds the one with .predict()
def load_trained_model():
    candidates = ["student_performance_prediction.pkl", "model.pkl"]
    
    # 1. Check known file names first
    for fname in candidates:
        if os.path.exists(fname):
            try:
                obj = joblib.load(fname)
                if hasattr(obj, "predict"):
                    return obj, fname
            except Exception:
                pass

    # 2. Search every .pkl file in current directory
    for fname in os.listdir("."):
        if fname.endswith(".pkl"):
            try:
                obj = joblib.load(fname)
                if hasattr(obj, "predict"):
                    return obj, fname
            except Exception:
                pass

    return None, None

model, model_filename = load_trained_model()

if model is None:
    st.error("Could not find a valid model file with a `.predict()` method. Ensure `student_performance_prediction.pkl` is in this folder.")
    st.stop()

st.title("🎓 Student Performance Predictor")
st.caption(f"Loaded model file: `{model_filename}`")
st.write("Enter student information below to predict academic performance.")

# Input fields
with st.form("prediction_form"):
    st.subheader("Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_hours = st.number_input(
            "Study Hours per Week",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.5
        )
        attendance_rate = st.slider(
            "Attendance Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )
        previous_grades = st.number_input(
            "Previous Grades",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

    with col2:
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=["No", "Yes"]
        )
        parent_education = st.selectbox(
            "Parent Education Level",
            options=["Associate", "Bachelor", "Doctorate", "High School", "Master"]
        )

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    # Build dictionary matching the 10 features expected by the model
    input_dict = {
        "cat__Participation in Extracurricular Activities_No": 1.0 if extracurricular == "No" else 0.0,
        "cat__Participation in Extracurricular Activities_Yes": 1.0 if extracurricular == "Yes" else 0.0,
        "cat__Parent Education Level_Associate": 1.0 if parent_education == "Associate" else 0.0,
        "cat__Parent Education Level_Bachelor": 1.0 if parent_education == "Bachelor" else 0.0,
        "cat__Parent Education Level_Doctorate": 1.0 if parent_education == "Doctorate" else 0.0,
        "cat__Parent Education Level_High School": 1.0 if parent_education == "High School" else 0.0,
        "cat__Parent Education Level_Master": 1.0 if parent_education == "Master" else 0.0,
        "remainder__Study Hours per Week": float(study_hours),
        "remainder__Attendance Rate": float(attendance_rate),
        "remainder__Previous Grades": float(previous_grades),
    }

    features_df = pd.DataFrame([input_dict])
    
    # Predict using the array values
    prediction = model.predict(features_df.values)[0]
    
    st.success(f"### Predicted Performance Score: {prediction:.2f}")