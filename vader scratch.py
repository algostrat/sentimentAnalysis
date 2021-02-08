from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scattergl


analyzer = SentimentIntensityAnalyzer()

df = pd.read_csv("C:\\Users\\xlerv_000\\OneDrive\\Code\\PycharmProjects\\sentimentAnalysis\\sent_scores.csv")

print(df.columns)

def sentiment_analyzer_scores(text):
    score = analyzer.polarity_scores(text) #vader
    print(text)
    print(score)


for i in range(5):
    opinion = sentiment_analyzer_scores(df['tweet'][i])
    print(opinion)


fig = px.scatter(df, x="polarity", y="subjectivity", hover_data=["tweet", "tweet_id"])
fig.show()