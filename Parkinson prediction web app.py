# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:50:33 2025

@author: palak yadav
"""
import numpy as np
import pickle
import streamlit as st

# Load the trained model and scaler
model = pickle.load(open("C:/train/parkinsons_model.sav", 'rb'))
scaler = pickle.load(open("C:/train/scaler.pkl", 'rb'))

# Prediction function
def parkinson_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        return "The Person does not have Parkinson's Disease"
    else:
        return "The Person has Parkinson's Disease"

# Main function for the Streamlit app
def main():
    st.title('Parkinson’s Disease Prediction Web App')

    # Collect input features
    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    jitter_percent = st.text_input('MDVP:Jitter(%)')
    jitter_abs = st.text_input('MDVP:Jitter(Abs)')
    rap = st.text_input('MDVP:RAP')
    ppq = st.text_input('MDVP:PPQ')
    ddp = st.text_input('Jitter:DDP')
    shimmer = st.text_input('MDVP:Shimmer')
    shimmer_db = st.text_input('MDVP:Shimmer(dB)')
    apq3 = st.text_input('Shimmer:APQ3')
    apq5 = st.text_input('Shimmer:APQ5')
    dda = st.text_input('Shimmer:DDA')
    nhr = st.text_input('NHR')
    hnr = st.text_input('HNR')
    rpde = st.text_input('RPDE')
    dfa = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    d2 = st.text_input('D2')
    ppe = st.text_input('PPE')

    diagnosis = ""

    if st.button("Parkinson's Test Result"):
        try:
            input_features = [float(x) for x in [fo, fhi, flo, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                                                 apq3, apq5, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]]
            diagnosis = parkinson_prediction(input_features)
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numbers for all fields.")

if __name__ == '__main__':
    main()
