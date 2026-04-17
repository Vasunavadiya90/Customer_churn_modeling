import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('One_hot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



## streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
credit_score = st.number_input('Credit Score')
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


# Combine one-hot encoded columns with input data (Geography features first)
input_data = pd.concat([geo_encoded_df, input_data.reset_index(drop=True)], axis=1)

# Reorder columns to match training data order
column_order = ['Geography_France', 'Geography_Germany', 'Geography_Spain', 
                'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
input_data = input_data[column_order]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
