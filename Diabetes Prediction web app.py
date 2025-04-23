# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:36:48 2025

@author: palak yadav
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open("C:/train/trained_model.sav",'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    #changing the input_data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'






def main():
    
    
    #giving a title 
    st.title('Diabetes Prediction Web App')
    
    
    #getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BloodPressure Value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of the Person')
    
    
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a abutton for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
     main()
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    