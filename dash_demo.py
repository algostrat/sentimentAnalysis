import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import collections
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_table

app = dash.Dash(__name__)

# change background color using hex
colors = {
    'background': '#2b3e50',
    'background2': '#000000',
}

"""
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
# word frequencies
counts = collections.Counter(noun_phrases)
trend_words = pd.DataFrame(counts.most_common(30),
                            columns=['words', 'count'])
"""

df = pd.read_pickle('data/sentdf.pkl')
trend_words = pd.read_pickle('data/trends.pkl')
abuse_df = pd.read_csv('data/abuse_df.csv')
abuse_df = abuse_df.loc[:, ~abuse_df.columns.str.contains('^Unnamed')]


#----------------------------------
fig1 = px.scatter(df, x='polarity', y='subjectivity', hover_data=["tweet", "tweet_id"])
fig2 = px.scatter(df, x="polarity", y="subjectivity", hover_data=["tweet", "tweet_id"])
fig3 = px.bar(trend_words[0:20], x='words', y='count')
fig4 = go.Figure()
fig5 = px.pie(trend_words[0:10], names='words', values='count')
fig6 = px.bar(trend_words[0:10], x='words', y='count')
fig7 = px.scatter()

# Change fig marker color

fig1.update_traces(marker=dict(size=9,
                               line=dict(width=2,
                                         color='#FFFFFF')))

fig2.update_traces(marker=dict(size=9,
                               line=dict(width=2,
                                         color='#FFFFFF')))

fig3.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(255,255,255)',
                   marker_line_width=1.5, opacity=1)

# Change fig background color
fig1.update_layout(
    font=dict(color='#FFFFFF'),
    plot_bgcolor=colors['background2'],
    paper_bgcolor=colors['background']
)

fig2.update_layout(
    font=dict(color='#FFFFFF'),
    plot_bgcolor=colors['background2'],
    paper_bgcolor=colors['background']
)

fig3.update_layout(
    font=dict(color='#FFFFFF'),
    plot_bgcolor=colors['background2'],
    paper_bgcolor=colors['background']
)
fig4.update_layout(
    font=dict(color='#FFFFFF'),
    plot_bgcolor=colors['background2'],
    paper_bgcolor=colors['background']
)
fig5.update_layout(
    font=dict(color='#FFFFFF'),
    plot_bgcolor=colors['background2'],
    paper_bgcolor=colors['background']
)
fig6.update_layout(
    font=dict(color='#FFFFFF'),
    plot_bgcolor=colors['background2'],
    paper_bgcolor=colors['background']
)
fig7.update_layout(
    font=dict(color='#FFFFFF'),
    plot_bgcolor=colors['background2'],
    paper_bgcolor=colors['background']
)
# tab

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'color': '#000000'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold'
}


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO],
                # mobile
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

# app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(
                [
                    dbc.CardImg(
                        src="assets/fau logo 2.png",
                        bottom=True, className="center"),
                ],
                style={"width": "8rem",
                       "height": "auto",
                       "margin-left": "auto",
                       "margin-right": "auto"
                       },
            )
        ])
    ]),

    dbc.Row([
        dbc.Col(
            html.H1("Twitter Sentiment Analysis",
                    className='text-center mb-4'),
            width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
                dcc.Tab(label='Members', value='tab-1', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Abstract', value='tab-2', style=tab_style, selected_style=tab_selected_style),
            ], className='text-center mb-2', style=tabs_styles),
            html.Div(id='tabs-content-inline')
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H2("Overall Sentiment Analysis",
                    className='text-center mb-4')
        ])
    ]),

    dbc.Row([
        # old graph info
        # dbc.Col([
        # html.H6("Plotting Vader's compound vs Textblob's subjectivity. "
        #       "compound: {negative <= 0.05, neutral >= 0.05 "
        #      "and <= 0.05, positive > 0.05}.",
        #     className='text-center mb-2'),
        # ], width={'size': 6}),

        dbc.Col([

            html.H6("Plotting Textblob subjectivity vs. polarity.",
                    className='text-center mb-2'),
        ], width={'size': 12}),

    ], justify='center'),

    dbc.Row([
        # old graph
        # dbc.Col([

        #   dcc.Graph(id='Graph1', figure=fig1)
        # ], width={'size': 6}),

        dbc.Col([

            dcc.Graph(id='Graph2', figure=fig2)
        ], width={'size': 12}),

    ], justify='center'),

    dbc.Row([
        dbc.Col([

            html.H2("Most Common Words",
                    className='text-center mb-2'),
            # add callback function here

            dcc.Graph(id='Graph3', figure=fig3)

        ])
    ], justify='center', className='text-center mb-5'),

    # top 10 words graphs
    dbc.Row([
        dbc.Col([

            html.H2("Top 10 Words",
                    className='text-center mb-2'),

        ])
    ]),

    dbc.Row([
        dbc.Col([

            dcc.Graph(id='Graph5', figure=fig5)
        ], width={'size': 6}),

        dbc.Col([

            dcc.Graph(id='Graph6', figure=fig6)
        ], width={'size': 6}),

    ], justify='center', className='text-center mb-5'),

    dbc.Row([
        dbc.Col([

            html.H2("Most Common Words v2",
                    className='text-center mb-5'),

            dcc.Dropdown(id='my_dropdown', multi=False,
                         options=[
                             {'label': trend_words['words'][0], 'value': trend_words['words'][0]},
                             {'label': trend_words['words'][1], 'value': trend_words['words'][1]},
                             {'label': trend_words['words'][2], 'value': trend_words['words'][2]},
                             {'label': trend_words['words'][3], 'value': trend_words['words'][3]},
                             {'label': trend_words['words'][4], 'value': trend_words['words'][4]},
                             {'label': trend_words['words'][5], 'value': trend_words['words'][5]},
                             {'label': trend_words['words'][6], 'value': trend_words['words'][6]},
                             {'label': trend_words['words'][7], 'value': trend_words['words'][7]},
                             {'label': trend_words['words'][8], 'value': trend_words['words'][8]},
                             {'label': trend_words['words'][9], 'value': trend_words['words'][9]},
                             {'label': '[View abusive tweets scores]', 'value': '_abusive_list_'}
                         ],
                         optionHeight=45,
                         disabled=False,
                         placeholder='Please select a word',
                         persistence=True,
                         persistence_type='memory',
                         style={"background-color": "#FFFFFF",
                                'color': '#000000'},

                         ),
            dcc.Graph(id='Graph4', figure=fig4)

        ])
    ]),

    # Abuse fig
    dbc.Row([
        dbc.Col([
            #deleted for now, we can add an abuse scatter plot later if needed
            #html.H2("Abuse figure",
          #          className='text-center mb-2'),

        ])
    ]),

    dbc.Row([
        dbc.Col([

          #  dcc.Graph(id='Graph7', figure=fig7)
        ]),
    ], justify='center'),

    dbc.Row([
        dbc.Col([
            html.H2("Abuse List",
                    className='text-center mb-4')
        ])
    ]),

    dbc.Row([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i}
                     for i in abuse_df.columns],
            data=abuse_df.to_dict('records'),
            style_cell=dict(textAlign='left'),
            style_header=dict(backgroundColor="Blue"),
            row_deletable=True,
            style_data=dict(backgroundColor="Black")
        )

    ], style={
        'padding': '20px',
        'width': '87%',
        'overflow-y': 'scroll',
        'height': '600px',
        'display': 'block',
        'textAlign': 'center',
        "margin-left": "auto",
        "margin-right": "auto"
        # persistance type for memory?
    }),

], fluid=True)


@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H5("Ryan Wahler, "
                    "Joshua Cidoni-Walker, "
                    "Muhammad Ahmed, "
                    "Ali Fatta, "
                    "Ali Khan",
                    className='text-center mb-5')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H5("Data sentiment analysis is used to determine if a post is positive or negative."
                    "For this project, we will be analyzing posts on social media such as Twitter "
                    "and determining if a post is abusive and if so, it will be flagged. "
                    "The web-based system will flag posts for human review any likely-abusive "
                    "posts, including those making personal attacks, inciting violence, "
                    "using abusive language, and/or generally making the internet a worse "
                    "place for others.",
                    className='text-center mb-5'),

        ])


@app.callback(
    Output(component_id='Graph4', component_property='figure'),
    Input(component_id='my_dropdown', component_property='value')
)

def update_graph(my_dropdown):
    print(type(my_dropdown))

    if my_dropdown is None:
        my_dropdown = ''

    if my_dropdown == '_abusive_list_':
        dff = abuse_df
    else:
        dff = df[df["tweet"].str.contains(my_dropdown)]

    scatterplot = px.scatter(
        data_frame=dff,
        x="polarity", y="subjectivity", hover_data=["tweet", "tweet_id"],
    )

    scatterplot.update_xaxes(range=[-1.1, 1.1])
    scatterplot.update_yaxes(range=[-0.1, 1.1])

    scatterplot.update_layout(
        font=dict(color='#FFFFFF'),
        plot_bgcolor=colors['background2'],
        paper_bgcolor=colors['background']
    )

    scatterplot.update_traces(marker=dict(size=9,
                                          line=dict(width=2,
                                                    color='#FFFFFF')))

    return (scatterplot)


if __name__ == '__main__':
    app.run_server(debug=True)
