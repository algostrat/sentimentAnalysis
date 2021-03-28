"""
line #54 is the part that applys abuse algo to every instance in dataframe
df['abuse_prediction'] = df['tweet'].apply(lambda x: predict(x, threshold=0.3))
All the preceeding code is just to recreate the df as it is in the main dash demo file
"""
import pandas as pd
import re
from notebook_abuse_model.abuse_model import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

df = pd.read_csv("sent_scores.csv")

analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(text):
    return analyzer.polarity_scores(text)

noun_phrases = []

df['nouns'] = df['nouns'].astype(object)


for i in range(len(df)):
    # sentiment_analyzer_scores(df['tweet'][i])
    tweet = df['tweet'][i]
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)

    for url in urls:
        tweet = tweet.replace(url, '')
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet).replace('RT ', '')
    tweet = re.sub(r'[^\w]', ' ', tweet)

    # too sensitive
    df.at[i, 'tweet'] = tweet
    # print(tweet)

    vader_opinion = sentiment_analyzer_scores(tweet)

    opinion = TextBlob(tweet)

    df.at[i, 'polarity'] = opinion.sentiment.polarity
    df.at[i, 'subjectivity'] = opinion.sentiment.subjectivity
    # print(vader_opinion['compound'])
    df.at[i, 'compound'] = vader_opinion['compound']
    df.at[i, 'nouns'] = opinion.noun_phrases

    # print(opinion.noun_phrases)
    for noun in opinion.noun_phrases:
        noun_phrases.append(noun.lower())
    # print(opinion.noun_phrases)


df['abuse_prediction'] = df['tweet'].apply(lambda x: predict(x, threshold=0.3))

#sample_df = df.loc[df['abuse_prediction'] == True]
df.to_csv('sent_scores1.csv')
print(df['abuse_prediction'])
#print(sample_df['tweet'])