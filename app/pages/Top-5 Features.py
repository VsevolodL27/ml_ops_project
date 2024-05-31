import json
import streamlit as st
import plotly.express as px 
from src.scorer import get_feature_importance

st.set_page_config(
    page_title='Топ-5 Features'
)

st.logo('./images/MTS_LOGO.png')

st.markdown("<h1 style='text-align: center;'>TOP-5 Features By Importance</h1>", unsafe_allow_html=True)

importance_features = get_feature_importance()
j = st.json(importance_features)

chart_data = { 'Feature': importance_features.keys(), 'Importance': importance_features.values() }
fig=px.bar(chart_data, x='Importance', y='Feature', orientation='h')
st.write(fig)

st.download_button('Download', json.dumps(importance_features, ensure_ascii=False), 'top-5-feature-importance.json')