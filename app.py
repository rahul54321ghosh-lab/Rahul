import streamlit as st
import pickle
import pandas as pd

# --- Load the trained model and its columns
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.markdown("<h1 style='color:#4CAF50;text-align:center;'>💼 Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("#### Enter your information and get an estimated salary below! 🚀")

# --- Layout with two columns
col1, col2 = st.columns(2)
with col1:
    experience = st.number_input("Years of Experience:", 0, 50, 1)
    age = st.number_input("Age:", 18, 100, 30)
with col2:
    location = st.selectbox("Location:", ['Urban', 'Suburban', 'Rural'])
    job_title = st.selectbox("Job Title:", ['Manager', 'Director', 'Analyst'])

# --- Prepare input for prediction
input_dict = {
    'Experience': experience,
    'Age': age,
    'Location': location,
    'Job_Title': job_title
}
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)

# --- Ensure all model columns are present
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# --- Predict and display result
if st.button("Predict Salary"):
    prediction = model.predict(input_encoded)
    st.success(f"💰 Estimated Salary: {prediction[0]:,.2f}")
    st.balloons()

# --- Extra footer for a professional look
st.markdown("---")
st.markdown("<p style='font-size:16px;text-align:center;'>Made with ❤ using Streamlit</p>", unsafe_allow_html=True)