# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:50:33 2025

@author: palak yadav
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open("C:/train/heart_disease_data (1).sav",'rb'))

#creating a function for prediction

def heart_prediction(input_data):
    
    input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
      return 'The Person does not have a Heart Disease'
    else:
      return 'The Person has Heart Disease'


def main():
    
    #giving a title
    st.title('Heart Prediction Web App')
    
    #giving the input data from the user
    
    
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Chest Pain Types')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestrol in mg/dl')
    fbs= st.text_input('Fasting Blood Sugar>120mg/dl')
    restecg = st.text_input('RestingElectrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST Depression Induced by Exercise')
    slopek= st.text_input('Slope Of The Peak exercise ST segment')
    ca= st.text_input('Major Vessels colored by Fluroscopy')
    thal= st.text_input('that:0=normal;1=fixed defect;2=reverssible defect')
    
    
    
    
    
    
    #code for prediction
    diagnosis=''
    
    
    #creating a button for prediction
    if  st.button('Heart Test Result'):
        diagnosis=heart_prediction_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
        st.success(diagnosis)
        
        
        
if __name__ == '__main__':
    main()
        