import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load and prepare the dataset
house_data = fetch_california_housing()
df = pd.DataFrame(house_data.data, columns=house_data.feature_names)
df['price'] = house_data.target

X = df.drop(['price'], axis=1)
Y = df['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Streamlit app UI
st.title("🏠 California House Price Predictor")
st.markdown("Enter the values for the features below to predict the house price:")

# Input form
MedInc = st.number_input('Median Income (10k USD)', value=3.0)
HouseAge = st.number_input('House Age', value=20)
AveRooms = st.number_input('Average Rooms', value=5.0)
AveBedrms = st.number_input('Average Bedrooms', value=1.0)
Population = st.number_input('Population', value=1000)
AveOccup = st.number_input('Average Occupancy', value=3.0)
Latitude = st.number_input('Latitude', value=34.0)
Longitude = st.number_input('Longitude', value=-118.0)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(input_data)
    st.success(f"🏡 Predicted House Price: ${prediction[0]*100000:.2f}")
