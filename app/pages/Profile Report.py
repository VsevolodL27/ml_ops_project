import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

st.set_page_config(
    page_title='Profile Report'
)

st.logo('./images/MTS_LOGO.png')

st.markdown("<h1 style='text-align: center;'>Profile Report On Used Features</h1>", unsafe_allow_html=True)

df = pd.read_csv('./train_data/train.csv')[[
    'объем_данных', 'сегмент_arpu', 'on_net',
    'секретный_скор', 'сумма', 'частота',
    'продукт_2', 'продукт_1', 'доход',
    'частота_пополнения', 'pack_freq'
]]

st.dataframe(df)
pr = ProfileReport(df, title='Report')


st_profile_report(pr)