import pandas as pd
import lightgbm
import plotly.express as px 
from collections import Counter

model = lightgbm.Booster(model_file='./models/lgbm_model.txt')

def make_prediction(dt, path_to_file):
    threshold = 0.32
    submission = pd.DataFrame({
        'client_id': pd.read_csv(path_to_file)['client_id'],
        'preds': (model.predict(dt) > threshold).astype(int),
        'proba': model.predict(dt)
    })
    return submission

def to_csv(submission):
    csv = submission.drop(['proba'], axis=1).to_csv(index=False).encode('utf-8')
    return csv

def get_feature_importance(top_count=5):
    features = dict(sorted(zip(model.feature_name(), model.feature_importance().tolist()), key=lambda x: x[1], reverse=True)[: top_count])
    return features

def barplot(submission):
    preds = Counter(submission['preds'])
    chart_data = { 'Label': preds.keys(), 'Count': preds.values() }
    fig=px.bar(chart_data, x='Label', y='Count')
    fig.update_layout(title_text='Label distribution')
    return fig

def histplot(submission):
    fig = px.histogram(submission, x="proba", nbins=25)
    fig.update_layout(title_text='Probability distribution')
    return fig