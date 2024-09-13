import streamlit as st
import pandas as pd
st.write('Hello, world!')

uploaded_file = st.file_uploader('csv',type=['csv'])
df = pd.read_csv(uploaded_file)
st.dataframe(df)