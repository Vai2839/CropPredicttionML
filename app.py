import streamlit as st
import pickle

import numpy as np
import pandas as pd


import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


st.title('Crop Recommendation')

n=st.number_input("N")
p=st.number_input('P')
k=st.number_input('K')
tem=st.number_input('Temperature')
hum=st.number_input('Humidty')
ph=st.number_input('ph')
rain=st.number_input('rainfall')
LabelMapping ={'apple': 0, 'banana': 1, 'blackgram': 2, 'chickpea': 3, 'coconut': 4, 'coffee': 5, 'cotton': 6, 'grapes': 7, 'jute': 8, 'kidneybeans': 9, 'lentil': 10, 'maize': 11, 'mango': 12, 'mothbeans': 13, 'mungbean': 14, 'muskmelon': 15, 'orange': 16, 'papaya': 17, 'pigeonpeas': 18, 'pomegranate': 19, 'rice': 20, 'watermelon': 21}
LabelMapping = {v: k for k, v in LabelMapping.items()}
submit=st.button('Submit')
if submit:
    features=np.array([[n,p,k,tem,hum,ph,rain]])
    pred=model.predict(features)
    if pred[0]in LabelMapping:
        st.write(LabelMapping[pred[0]])

