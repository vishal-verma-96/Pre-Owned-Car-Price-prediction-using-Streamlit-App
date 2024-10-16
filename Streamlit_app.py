import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the model
loaded_model = pickle.load(open('app.pkl', 'rb'))

# Load the cleaned data
cleaned_data = pd.read_csv('Car_Details_Cleaned_Dataset.csv')

# Define categorical columns
category_col = ['Car_Brand', 'Car_Name', 'Fuel', 'Seller_Type', 'Transmission', 'Owner']

# Function for encoding data
def preprocess_data(df, label_encoders):
    for feature in df.columns:
        if feature in label_encoders:
            df[feature] = label_encoders[feature].transform(df[feature])
    return df

# Load the LabelEncoders used during training
label_encoders = {}
for feature in category_col:
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_data[feature])
    label_encoders[feature] = label_encoder

# Add CSS for background image
image_path = "https://raw.githubusercontent.com/vishal-verma-96/Capstone_Project_By_Skill_Academy/main/Header_image.jpg"
st.markdown(
    f"""
    <style>
    .header {{
        background-image: url('{image_path}');
        background-size: cover;
        background-position: center;
        height: 150px; /* Adjust height as needed */
        opacity: 0.3; /* Set opacity for transparency */
        position: relative; /* To position child elements absolutely */
        z-index: -1; /* Ensure it stays behind other elements */
    }}
    .header h1, .header h2 {{
        color: white; /* Change text color to white for visibility */
        position: absolute; /* Position text absolutely */
        left: 50%; /* Center horizontally */
        transform: translateX(-50%); /* Adjust to center */
    }}
    </style>
    <div class="header">
        <h1>Car Selling Price Prediction App</h1>
        <h2>Please provide the required details to predict the car's selling price.</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Providing Sidebar
st.sidebar.markdown("""
This application predicts the selling price of a car based on various features.
### How to use:
1. **Select the Car Details:** Select the correct input of car characteristics from the provided options.
2. **Predict Price:** Click on the 'Predict Selling Price' button to see the predicted price.
""")

# Encode the loaded dataset
encoded_data = preprocess_data(cleaned_data.copy(), label_encoders)

# Display sliders for numerical features
km_driven = st.slider("Select KM Driven:", min_value=int(cleaned_data["Km_Driven"].min()),
                      max_value=int(cleaned_data["Km_Driven"].max()))
year = st.slider("Select Year:", min_value=int(cleaned_data["Year"].min()), max_value=int(cleaned_data["Year"].max()))

# Display dropdowns for categorical features
selected_brand = st.selectbox("Select Car Brand:", cleaned_data["Car_Brand"].unique())
brand_filtered_df = cleaned_data[cleaned_data['Car_Brand'] == selected_brand]
selected_model = st.selectbox("Select Car Model:", brand_filtered_df["Car_Name"].unique())
selected_fuel = st.selectbox("Select Fuel:", cleaned_data["Fuel"].unique())
selected_seller_type = st.radio("Select Seller Type:", cleaned_data["Seller_Type"].unique())
selected_transmission = st.radio("Select Transmission:", cleaned_data["Transmission"].unique())
selected_owner = st.selectbox("Select Owner:", cleaned_data["Owner"].unique())

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({'Car_Brand': [selected_brand],
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

# Standardize numerical features using scikit-learn's StandardScaler
scaler = StandardScaler()
numerical_cols = ['Year', 'Km_Driven']
input_data_encoded[numerical_cols] = scaler.fit_transform(input_data_encoded[numerical_cols])

# Make prediction using the loaded model
if st.button("Predict Selling Price"):
    # Make predictions
    predicted_price = loaded_model.predict(input_data_encoded)
    st.subheader("Predicted Selling Price:")
    st.write(f"The predicted selling price is: **_{predicted_price[0]:,.2f}_**")
