# **Data-Science-Capstone-Project**
### Description:
This repository contains the final capstone project for the Data Science with Python Career Program at Skill Academy. The project focuses on predicting the selling prices of pre-owned cars using a dataset of car details. It includes exploratory data analysis (EDA), training and evaluating predictive models and deploying a user-friendly web application using Streamlit. Users can input car details on the app, and the predicted selling price will be displayed on the interface.

<img src = "https://github.com/vishal-verma-96/Capstone_Project_By_Skill_Academy/blob/501dbfa3ffde34144dbf8262f5ec2b39e5f07f82/Readme_image.jpg">

### Project Goals:
* **Explore and Analyze:** Conduct an in-depth exploration of the car dataset to understand its characteristics and identify potential relationships between features.
* **Visualize:** Create insightful data visualizations to uncover patterns, trends, and anomalies.
* **Preprocess and Clean:** Address duplicate values, outliers, inconsistencies, and other data quality issues to prepare the data for modeling.
* **Machine Learning Model Development:** Experiment with various machine learning techniques.
* **Model Evaluation:** Employ metrics to assess model performance, compare different approaches, and select the best-performing model.
* **Sample Prediction:** Demonstrate the functionality of the final model by generating predictions on sample data.
* **Streamlit Deployment:** Develop a user-friendly web application using Streamlit for real-time car price predictions based on user-provided parameters.

### Dataset Preview:
A preview of the top five rows of the original or raw dataset.

| | name | year | selling_price | kms_driven | fuel | seller_type | transmission | owner |
|-| ---------------------------- | ---- | ------------- | ---------- | ---- | ----------- | ------------ | ----- |
|0| Maruti 800 AC |	2007 | 60000 | 70000 | Petrol | Individual | Manual | First Owner
|1|	Maruti Wagon R LXI Minor | 2007 | 135000 | 50000 | Petrol | Individual | Manual | First Owner
|2|	Hyundai Verna 1.6 SX | 2012 | 600000 | 100000	| Diesel | Individual | Manual | First Owner
|3|	Datsun RediGO T Option | 2017 | 250000 | 46000 | Petrol	| Individual | Manual	| First Owner
|4|	Honda Amaze VX i-DTEC | 2014 | 450000	| 141000 | Diesel	| Individual | Manual	| Second Owner

### Description of features of the dataset:
The describing the features of raw dataset, which were shown above, are as follows:

```Car_Name:``` Name of Car sold

```Year:``` Year in which the car was bought from the showroom (means by the car company, not by any seller)

```Selling_Price:``` Price at which car sold

```Kms_Driven:``` Number of Kilometers Car driven before it is sold

```Fuel_Type:``` Type of fuel Car uses

```Seller_Type:``` Type of seller 

```Transmission:``` Gear transmission of the car (Automatic / Manual)

```Owner:``` Number of previous owners 

# Technologies Used:
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle
* Streamlit
