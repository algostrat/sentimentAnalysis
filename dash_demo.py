import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import collections
#import dash_table
#import plotly.graph_objects as go

#import wordcloud
#from autocorrect import Speller

app = dash.Dash(__name__)

df = pd.read_csv("sent_scores.csv")

analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(text):
    return analyzer.polarity_scores(text)

# create a list a column in dataframe to store all nouns for each tweet
noun_phrases = []
df['nouns'] = df['nouns'].astype(object)

for i in range(len(df)):
    #sentiment_analyzer_scores(df['tweet'][i])
    tweet = df['tweet'][i]
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)

    # removing links and excess whitespaces
    for url in urls:
        tweet = tweet.replace(url, '')
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet).replace('RT ','')
    tweet = re.sub(r'[^\w]', ' ', tweet)

    df.at[i,'tweet'] = tweet
    #print(tweet)

    # create score from vader and textblob
    vader_opinion = sentiment_analyzer_scores(tweet)
    opinion = TextBlob(tweet)

    # add scores to dataframe from textblob
    df.at[i, 'polarity'] = opinion.sentiment.polarity
    df.at[i, 'subjectivity'] = opinion.sentiment.subjectivity

    # add scores to dataframe from vader
    #print(vader_opinion['compound'])
    df.at[i, 'compound'] = vader_opinion['compound']
    df.at[i, 'nouns'] = opinion.noun_phrases

    # get noun words from textblob
    for noun in opinion.noun_phrases:
        noun_phrases.append(noun.lower())
    #print(opinion.noun_phrases)

#print(df)


# generate word frequencies from noun words
counts = collections.Counter(noun_phrases)
clean_tweets = pd.DataFrame(counts.most_common(30),
                             columns=['words', 'count'])

#print(clean_tweets)

# create plotly figures
fig1 = px.scatter(df, x="compound", y="subjectivity", hover_data=["tweet", "tweet_id"])
fig2 = px.scatter(df, x="polarity", y="subjectivity", hover_data=["tweet", "tweet_id"])
fig3 = px.bar(clean_tweets, x='words', y='count')


#--------Preliminary abuse stuff - to be deleted
bad_tweets = []
tweet_id = []
for i in range(len(df)):
    if df['compound'][i] < -0.3 or df['polarity'][i] < -0.6:
        bad_tweets.append(df['tweet'][i][0:30])
        tweet_id.append(df['tweet_id'][i])
# --- ignore above


my_df = pd.DataFrame(dict(col1=bad_tweets, col2=tweet_id))
my_df = clean_tweets[0:10]

print(my_df['words'])

#-----------------

# begin plotting plotly figures to Dash front end dashboard
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

""" # extra figure

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


#@app.callback(

# )

if __name__ == '__main__':
    app.run_server(debug=True)