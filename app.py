import streamlit as st
import pandas as pd
import joblib

def load_model():
    return joblib.load("linear_regression_model.pkl")

def predict(avg_session_length, time_on_app, time_on_website, length_of_membership):
    model = load_model()
    user_data = pd.DataFrame({
        'Avg. Session Length': [avg_session_length],
        'Time on App': [time_on_app],
        'Time on Website': [time_on_website],
        'Length of Membership': [length_of_membership]
    })
    prediction = model.predict(user_data)
    return prediction[0]

# Streamlit UI
st.title("Ecommerce Clothes Sales Prediction")
st.write("Predict the yearly amount spent by customers based on their usage patterns.")

# Input fields
avg_session_length = st.number_input("Average Session Length", min_value=0.0, step=0.1)
time_on_app = st.number_input("Time on App", min_value=0.0, step=0.1)
time_on_website = st.number_input("Time on Website", min_value=0.0, step=0.1)
length_of_membership = st.number_input("Length of Membership", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict Yearly Amount Spent"):
    prediction = predict(avg_session_length, time_on_app, time_on_website, length_of_membership)
    st.success(f"Predicted Yearly Amount Spent: ${prediction:.2f}")