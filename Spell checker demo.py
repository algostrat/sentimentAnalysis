import nltk
from nltk.collocations import *
from nltk.corpus import stopwords

import numpy as np

import plotly.express as px
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re


import collections
#import dash_table
#import plotly.graph_objects as go
#import wordcloud
from autocorrect import Speller

spell = Speller(lang='en')

df = pd.read_csv("sent_scores.csv")

analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(text):
    return analyzer.polarity_scores(text)

noun_phrases = []

df['nouns'] = df['nouns'].astype(object)

print(spell("helllo mye name is youre"))

for i in range(len(df)):
    #sentiment_analyzer_scores(df['tweet'][i])
    tweet = df['tweet'][i]
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)

    for url in urls:
        tweet = tweet.replace(url, '')
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet).replace('RT ','')
    tweet = re.sub(r'[^\w]', ' ', tweet)
    tweet = re.sub(' +', ' ', tweet)

    tweet = tweet.lower()

    df.at[i,'tweet'] = tweet

    vader_opinion = sentiment_analyzer_scores(tweet)
    opinion = TextBlob(tweet)

    df.at[i, 'polarity'] = opinion.sentiment.polarity
    df.at[i, 'subjectivity'] = opinion.sentiment.subjectivity
    # print(vader_opinion['compound'])
    df.at[i, 'compound'] = vader_opinion['compound']
    df.at[i, 'nouns'] = opinion.noun_phrases

    nouns = opinion.noun_phrases #not accurate

    token = nltk.wordpunct_tokenize(tweet)

    print(token)

    #spell checking
    for word in token:
        print(spell(word))

    if i > 2:
        break
