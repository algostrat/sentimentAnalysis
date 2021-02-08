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

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#nltk.download()

df = pd.read_csv("sent_scores.csv")

#x = nltk.corpus.genesis.words('english-web.txt')
# text_entries = df.to_numpy()
# np.savetxt("text_file.txt", text_entries, fmt = "%d")


analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(text):
    return analyzer.polarity_scores(text)

noun_phrases = []

df['nouns'] = df['nouns'].astype(object)

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
    tweet = spell(tweet)

    df.at[i,'tweet'] = tweet

    vader_opinion = sentiment_analyzer_scores(tweet)
    opinion = TextBlob(tweet)

    df.at[i, 'polarity'] = opinion.sentiment.polarity
    df.at[i, 'subjectivity'] = opinion.sentiment.subjectivity
    # print(vader_opinion['compound'])
    df.at[i, 'compound'] = vader_opinion['compound']
    df.at[i, 'nouns'] = opinion.noun_phrases

    nouns = opinion.noun_phrases #not accurate

    # preliminary trend finder
    for noun in opinion.noun_phrases:
        noun_phrases.append(noun.lower())

    #removing stop words
    word_tokens = nltk.word_tokenize(tweet, False)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    print(filtered_sentence)

    # optimized trend finder----------------------------------------
    # Using NLTK library to finding common 2 and 3 word phrases.
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    # Reading in data saved for Phrase frequencies on line 22 and 23.
    # finder = BigramCollocationFinder.from_words(
    #     nltk.corpus.genesis.words('text_file.txt'))

    token = nltk.wordpunct_tokenize(tweet)
    print("tweet: "+ tweet)
    #print("token :"+str(token))

    finder = TrigramCollocationFinder.from_words(token)
    print(finder.nbest(bigram_measures.pmi, 4),"\n")

    # Keeping only bigrams that more than 8 times, depending on the size of datasets we can adjust the number.
    finder.apply_freq_filter(8)

    # Lastly, we return, print, the 10 n-grams with the highest PMI (Pointwise Mutual Information)
    finder.nbest(bigram_measures.pmi, 10)



    #because this is running as a script only test on small set
    if i > 3:
        break

