import pandas_profiling
import nltk
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sb
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
import unidecode
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
from nltk.stem import PorterStemmer
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import matplotlib.animation as animation
import operator
import plotly.express as px
from collections import Counter

df = pd.read_csv('data\\train_E6oV3lV.csv')
print(df.head())
df.drop_duplicates(inplace = True)

short_df = df[:][0:2]
print(short_df)

#Code to remove @
df['clean_tweet'] = df['tweet'].apply(lambda x : ' '.join([tweet for tweet in x.split()if
                                                           not tweet.startswith("@")]))
#Removing numbers
df['clean_tweet'] = df['clean_tweet'].apply(lambda x
                                            : ' '.join([tweet for tweet in x.split() if not tweet == '\d*']))

#Removing all the greek characters using unidecode library
df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([unidecode.unidecode(word) for word in x.split()]))

#Spellcheck and change every word
#df['clean_tweet'] = df['clean_tweet'].apply(lambda x : ' '.join([tweet for tweet in]))

short_df['clean_tweet'] = short_df['clean_tweet'].apply(lambda x : print([tweet for tweet in x]))

print(short_df)

# token = nltk.wordpunct_tokenize(tweet)
# spell checking
#for word in token:
#    print(spell(word))