import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🏦 Customer Churn Prediction (ANN)")

# -------------------------------------------------
# Cache Model Training (IMPORTANT FOR SPEED)
# -------------------------------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("Churn_Modelling.csv")

    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # One Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build ANN
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train model (only once because of caching)
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

    # Calculate accuracy
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, X.columns, acc


# Load trained model
model, scaler, columns, accuracy = train_model()

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("Enter Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
age = st.sidebar.number_input("Age", 18, 100, 35)
balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.sidebar.number_input("Number of Products", 1, 4, 1)
has_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 500000.0, 60000.0)

# Convert categorical to numeric
has_card = 1 if has_card == "Yes" else 0


# Create input dataframe
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_card],
    "EstimatedSalary": [salary]
})

# Reorder columns correctly
input_data = input_data.reindex(columns=columns, fill_value=0)

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.button("🔍 Predict"):

    input_scaled = scaler.transform(input_data)

    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0

    if prediction == 1:
        st.error("❌ Customer Will Exit")
    else:
        st.success("✅ Customer Will Stay")

    st.metric("Exit Probability", f"{prediction_prob*100:.2f}%")

# Show model accuracy
st.info(f"Model Accuracy: {accuracy*100:.2f}%")
