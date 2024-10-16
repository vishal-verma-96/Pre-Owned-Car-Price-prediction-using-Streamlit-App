import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the model
loaded_model = pickle.load(open('app.pkl', 'rb'))

# Load the cleaned data
cleaned_data = pd.read_csv('Car_Details_Cleaned_Dataset.csv')

# Define categorical columns
categorical_col = ['Car_Brand', 'Car_Name', 'Fuel', 'Seller_Type', 'Transmission', 'Owner']

# Load the LabelEncoders used during training
label_encoders = {}
for feature in categorical_col:
    label_encoders[feature] = pickle.load(open(f'{feature}_label_encoder.pkl', 'rb'))

# Load the pre-fitted StandardScaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Function for encoding data
def preprocess_data(df, label_encoders):
    for feature in df.columns:
        if feature in label_encoders:
            df[feature] = label_encoders[feature].transform(df[feature])
    return df

# Title of the app
st.title("Car Selling Price Prediction App")
st.subheader("Please provide the required details to predict the car's selling price.")

st.sidebar.markdown("""
This application predicts the selling price of a car based on various features.
### How to use:
1. **Select the Car Details:** Use the sliders and dropdowns to input the car details.
2. **Predict Price:** Click on the 'Predict Selling Price' button to see the predicted price.
""")

# Display options for data
display_option = st.radio("Select Display Option:", ["No Data", "Loaded CSV Data", "Encoded Data"])

# Encode the loaded dataset
encoded_data = preprocess_data(cleaned_data.copy(), label_encoders)

# Display the selected data
if display_option == "Loaded CSV Data":
    st.subheader("Loaded CSV Data:")
    st.write(cleaned_data)
elif display_option == "Encoded Data":
    st.subheader("Encoded Data:")
    st.write(encoded_data)

# Display sliders for numerical features
km_driven = st.slider("Select KM Driven:", min_value=int(cleaned_data["Km_Driven"].min()),
                      max_value=int(cleaned_data["Km_Driven"].max()))
year = st.slider("Select Year:", min_value=int(cleaned_data["Year"].min()), max_value=int(cleaned_data["Year"].max()))

# Display dropdowns for categorical features
selected_brand = st.selectbox("Select Brand:", cleaned_data["Car_Brand"].unique())
brand_filtered_df = cleaned_data[cleaned_data['Car_Brand'] == selected_brand]
selected_model = st.selectbox("Select Model:", brand_filtered_df["Car_Name"].unique())
selected_fuel = st.selectbox("Select Fuel:", cleaned_data["Fuel"].unique())
selected_seller_type = st.selectbox("Select Seller Type:", cleaned_data["Seller_Type"].unique())
selected_transmission = st.selectbox("Select Transmission:", cleaned_data["Transmission"].unique())
selected_owner = st.selectbox("Select Owner:", cleaned_data["Owner"].unique())

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'Car_Brand': [selected_brand],
    'Car_Name': [selected_model],
    'Year': [year],
    'Km_Driven': [km_driven],
    'Fuel': [selected_fuel],
    'Seller_Type': [selected_seller_type],
    'Transmission': [selected_transmission],
    'Owner': [selected_owner]
})

# Preprocess the user input data using the same label encoders
input_data_encoded = preprocess_data(input_data.copy(), label_encoders)

# Standardize numerical features using the pre-fitted StandardScaler
input_data_encoded[numerical_cols] = scaler.transform(input_data_encoded[numerical_cols])

# Display processed input data
st.subheader("Processed Input Data:")
st.write(input_data_encoded)

# Make prediction using the loaded model
if st.button("Predict Selling Price"):
    # Make predictions
    predicted_price = loaded_model.predict(input_data_encoded)
    st.subheader("Predicted Selling Price:")
    st.write(f"The predicted selling price is: **_{predicted_price[0]:,.2f}_**")
