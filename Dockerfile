FROM python:3.11.9
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY /app /app
EXPOSE 3001
CMD streamlit run Prediction.py --server.port 3001