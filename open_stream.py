import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


#load the trained model
model = tf.keras.models.load_model('model.h5')

#load the enocder and scaler
with open('one_hot_encode.pk1', 'rb') as file:
    one_hot_encoder = pickle.load(file)
    
with open('label_encoder_gender.pk1', 'rb') as file:
    label_encoder_gender  = pickle.load(file)
    
with open('Scaler.pk1', 'rb') as file:
    scaler = pickle.load(file)
    
    
#streamlit app
st.title('Customer Churn Prediction')

#user input
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has creditcard',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#prepare the input_data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumofProducts": [num_of_products],
    "Hascrcard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Estimated Salary": [estimated_salary]
})


#oneHot encoded 'Geography'
geo_encoded = one_hot_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))


#combined one hot encoded data with user_input
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the input data
scaler_data = scaler.transform(input_data.values)


#prediction churn
prediction = model.predict(scaler_data)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("The Customer is not likely to churn")
    




