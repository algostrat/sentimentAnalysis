import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import autocorrect

df = pd.read_csv("C:\\Users\\xlerv_000\\OneDrive\\Code\\PycharmProjects\\sentimentAnalysis\\sent_scores.csv")

analyzer = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(text):
    return analyzer.polarity_scores(text)

tweets = []

for i in range(10):
    #sentiment_analyzer_scores(df['tweet'][i])
    tweet = df['tweet'][i]
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)

    #remove urls/links
    for url in urls:
        tweet = tweet.replace(url, '')

    #remove mentions
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)

    tweets.append(tweet)

    opinion = TextBlob(tweet)
    vader_opinion = sentiment_analyzer_scores(tweet)
    df.at[i, 'polarity'] = opinion.sentiment.polarity
    df.at[i, 'subjectivity'] = opinion.sentiment.subjectivity
    #print(vader_opinion['compound'])
    df.at[i, 'compound'] = vader_opinion['compound']
    #df.at[i, 'nouns'] = opinion.noun_phrases

    #remove hashtags
    print(opinion.noun_phrases)


#sentiment scores
#for i in range(len(10)):





#remove and create a column for hashtags
#df['hashtag'] = df['tweet'].apply(lambda x: re.findall(r'\B#\w*[a-zA-Z]+\w*', x))

df.to_pickle("preproc_data.pkl")