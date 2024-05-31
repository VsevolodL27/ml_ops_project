import datetime
import tempfile
import streamlit as st
import src.preprocessing as preprocessing
import src.scorer as scorer

csv = None
st.set_page_config(
    page_title='MTS MLOps Homework'
)

st.markdown("<h1 style='text-align: center;'>MTS MLOps. Homework 1</h1>", unsafe_allow_html=True)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('./images/MTS_TETA_LOGO.jpg')
    
st.markdown("<h2 style='text-align: center;'>Churn Prediction</h2>", unsafe_allow_html=True)

st.logo('./images/MTS_LOGO.png')

uploaded_file = st.file_uploader(' ', 'csv', label_visibility='hidden')

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
        if st.button('Predict'):
            with st.spinner('File processing...'):
                data = uploaded_file.getvalue()
                f.write(data)
                filename = f.name

            with st.spinner('Data preprocessing...'):
                input_df = preprocessing.import_data(filename)
                preprocessed_df = preprocessing.run_preprocessing(input_df)

            with st.spinner('Make prediction...'):
                submission = scorer.make_prediction(preprocessed_df, filename)
                csv = scorer.to_csv(submission)

            with st.spinner('Label distribution'):
                fig = scorer.barplot(submission)
                st.plotly_chart(fig)

            with st.spinner('Probability distribution'):
                fig = scorer.histplot(submission)
                st.plotly_chart(fig)
            
            with st.spinner('Prediction dataframe'):
                st.dataframe(submission)

            st.download_button(
                label='Download file with prediction',
                data=csv,
                file_name='predict_' + datetime.datetime.now().isoformat() + '.csv',
                mime='text/csv'
            )