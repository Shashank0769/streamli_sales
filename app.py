import streamlit as st
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Load the trained model
model = pickle.load(open("sales_forecasting_model.pkl", "rb"))

# Get expected feature names from the model
expected_features = model.feature_names_in_

# Title
st.title("Sales Forecasting App")

# User input form
st.sidebar.header("Input Features")

def user_input_features():
    Store = st.sidebar.number_input("Store ID", min_value=1, step=1)
    DayOfWeek = st.sidebar.selectbox("Day of the Week", list(range(1, 8)))
    Open = st.sidebar.selectbox("Store Open", [0, 1])
    Promo = st.sidebar.selectbox("Promo Active", [0, 1])
    StateHoliday = st.sidebar.selectbox("State Holiday", ["0", "a", "b", "c"])
    SchoolHoliday = st.sidebar.selectbox("School Holiday", [0, 1])
    
    # Create DataFrame
    df = pd.DataFrame([[Store, DayOfWeek, Open, Promo, StateHoliday, SchoolHoliday]], 
                      columns=["Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"])
    
    return df

input_data = user_input_features()

# One-hot encode categorical variables
input_data = pd.get_dummies(input_data)


# Ensure all expected features exist
for feature in expected_features:
    if feature not in input_data.columns:
        input_data[feature] = 0  # Fill missing features with 0

# Reorder columns to match model's training data
input_data = input_data[expected_features]

# Mock past sales data for comparison
past_sales = np.random.randint(500, 1500, size=10)
past_dates = pd.date_range(end=pd.Timestamp.today(), periods=10)

# Make prediction
if st.button("Predict Sales"):
    prediction = model.predict(input_data)

    # Success message
    st.success("Sales prediction successful!")

    # Display predicted sales
    st.write(f"Predicted Sales: **${np.round(prediction[0], 2)}**")

    # Graphical Representation
    st.subheader("Past Sales Trends")
    fig, ax = plt.subplots()
    ax.plot(past_dates, past_sales, label='Past Sales')
    ax.axhline(y=prediction[0], color='r', linestyle='--', label='Predicted Sales')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Predicted Sales Comparison
    avg_past_sales = np.mean(past_sales)
    comparison = "higher" if prediction[0] > avg_past_sales else "lower"
    st.write(f"The predicted sales are {comparison} than the average past sales of **${np.round(avg_past_sales, 2)}**.")