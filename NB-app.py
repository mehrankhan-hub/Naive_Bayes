import pandas as pd
import streamlit as st
import pickle as pkl

with open('Naive_Bayes.pkl', 'rb') as f:
    mdl = pkl.load(f)

df= pd.read_csv('Naive-Bayes-Classification-Data.csv')
#print(df.head())

st.title('Diabetes Prediction System')
st.image('NB_image.jpg')
st.header('Predict whether a person has diabetes based on Glucose and Blood Pressure levels.')
glucose = st.slider('Glucose (Range 20 - 160)',max_value=160,min_value=20)
bloodpressure = st.slider('BloodPressure (Range 40 - 190)',max_value=190,min_value=40)

x_values = pd.DataFrame({
    'glucose':[glucose],'bloodpressure':[bloodpressure]
})

if st.button('Predict'):
    prediction = mdl.predict(x_values)
    if prediction == 0:
        st.success('No Diabetes Detected')
    else:
        st.error('Diabetes Detected')
st.table(x_values)




def footer():
    st.write("---")
    st.write("Heart Disease Prediction System")
    st.write("Developed by MEHRAN KHAN")
    st.write("© 2026 All Rights Reserved")
footer()

st.markdown("""
<style>

.stApp{
background:
linear-gradient(rgba(8,47,73,.75),rgba(8,47,73,.75)),
url("https://images.unsplash.com/photo-1538108149393-fbbd81895907?auto=format&fit=crop&w=1920&q=80");

background-size:cover;
background-position:center;
background-attachment:fixed;
}

.block-container{
background:rgba(255,255,255,.83);
padding:45px;
border-radius:22px;
box-shadow:0 15px 40px rgba(0,0,0,.55);
}

</style>
""", unsafe_allow_html=True)