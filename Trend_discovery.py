import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import collections
import dash_table
import plotly.graph_objects as go
import nltk
from nltk.collocations import *
import wordcloud
from autocorrect import Speller


app = dash.Dash(__name__)

df = pd.read_csv("sent_scores.csv")

text_entries = a_dataframe.to_numpy()
np.savetxt("text_file.txt", text_entries, fmt = "%d")

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

    #too sensitive
    df.at[i,'tweet'] = tweet
    #print(tweet)

    vader_opinion = sentiment_analyzer_scores(tweet)

    opinion = TextBlob(tweet)

    df.at[i, 'polarity'] = opinion.sentiment.polarity
    df.at[i, 'subjectivity'] = opinion.sentiment.subjectivity
    #print(vader_opinion['compound'])
    df.at[i, 'compound'] = vader_opinion['compound']
    df.at[i, 'nouns'] = opinion.noun_phrases

    for noun in opinion.noun_phrases:
        noun_phrases.append(noun.lower())
    #print(opinion.noun_phrases)

#Word frequencies (Trend Discovery)
counts = collections.Counter(noun_phrases)
clean_tweets = pd.DataFrame(counts.most_common(30),
                             columns=['words', 'count'])
fig3 = px.bar(clean_tweets, x='words', y='count')


fig1 = px.scatter(df, x="compound", y="subjectivity", hover_data=["tweet", "tweet_id"])
fig2 = px.scatter(df, x="polarity", y="subjectivity", hover_data=["tweet", "tweet_id"])

bad_tweets = []
tweet_id = []
for i in range(len(df)):
    if df['compound'][i] < -0.3 or df['polarity'][i] < -0.6:
        bad_tweets.append(df['tweet'][i][0:30])
        tweet_id.append(df['tweet_id'][i])

my_df = pd.DataFrame(dict(col1=bad_tweets, col2=tweet_id))

print(my_df)

# Phrase frequencies (Trend Discovery)

# Using NLTK library to finding common 2 and 3 word phrases.
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# Reading in data saved for Phrase frequencies on line 22 and 23. 
finder = BigramCollocationFinder.from_words(
    nltk.corpus.genesis.words('text_file.txt'))

# Keeping only bigrams that more than 8 times, depending on the size of datasets we can adjust the number.
finder.apply_freq_filter(8)

# Lastly, we return, print, the 10 n-grams with the highest PMI (Pointwise Mutual Information)
finder.nbest(bigram_measures.pmi, 10)



app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.H1(children='Overall Sentiment'),

        html.Div(children='''
            Plotting Vader's compound vs Textblob's subjectivity. compound: {negative <= 0.05,
            neutral >= 0.05 and <= 0.05, positive > 0.05}.
        '''),

        dcc.Graph(
            id='graph1',
            figure=fig1
        ),
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Overall Sentiment'),

        html.Div(children='''
        Plotting Textblob subjectivity vs. polarity.
    '''),

        dcc.Graph(
            id='graph2',
            figure=fig2
        ),
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Word Frequencies'),

        html.Div(children='''
            Most common words
        '''),

        dcc.Graph(
            id='graph3',
            figure=fig3
        ),
    ]),

    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Phrase Frequencies'),

        html.Div(children='''
            Most common phrases
        '''),

        dcc.Graph(
            id='graph4',
            figure=fig4
        ),
    ]),
"""
    html.Div(
        html.H1(children='Possible abusive instances'),
        html.Div(children='''
            curating list of abusive instances from vader and texblob scoring
        '''),
        dash_table.DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in my_df.columns],
            data = my_df.to_dict('records'),
        )
    )
"""

])

if __name__ == '__main__':
    app.run_server(debug=True)