"""
This file preprocesses the data and creates dataframe for sentiment scores and trends

1st create sent_scores dataframe and save it

"""
import pandas as pd
import numpy as np
import unidecode
import re
import pkg_resources
from symspellpy import SymSpell, Verbosity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from itertools import chain
from textblob import TextBlob
import pickle
import collections

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)

#default smaller dictionary
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")

bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

analyzer = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(text):
    return analyzer.polarity_scores(text)


# use a pickled time to check if csv file dates in tweet.csv are new or old
curr_time = pd.Timestamp.now()

col_names = ['raw_text', 'clean_text', 'url', 'username',
             'hashtag', 'created_time', 'retweet_count', 'followers_count', 'relevancy_score']

tweets = pd.read_csv('Tweets.csv', names=col_names)
tweets['created_time'] = pd.to_datetime(tweets['created_time'], unit='s')

#delete this when we get tweet.csv reader to work
tweets = pd.read_csv('../legacy_old_stuff/egn.csv')

sentdf = tweets

def preproc(s):
    """
    create abuse_df.csv
    :param s:
    :return:
    """
    # Removing all the greek characters using unidecode library
    s = ' '.join([unidecode.unidecode(word) for word in s.split()])

    # removing extra spaces/tabs
    s = ' '.join(s.split())

    # Code for removing slang words
    d = {'luv': 'love', 'wud': 'would', 'lyk': 'like', 'wateva': 'whatever', 'ttyl': 'talk to you later',
         'kul': 'cool', 'fyn': 'fine', 'omg': 'oh my god!', 'fam': 'family', 'bruh': 'brother',
         'cud': 'could', 'fud': 'food', 'lol': 'laugh out loud', 'wtf': 'what the fuck', 'wyd': 'what are you doing',
         'wdym': 'what do you mean', 'lmao': 'laugh my ass off', 'fml': 'fuck my life', 'np': 'no problem',
         'ffs': 'for fucks sake', 'nvm': 'nevermind', 'bro': 'brother', 'bra': 'brother',
         'tldr': 'too long, didn\'t read',
         'stfu': 'shut the fuck up', 'tbh': 'to be honest', 'idek': 'i don\'t even know',
         'diy': 'Do it yourself', 'rn': 'right now', 'btw': 'by the way', 'u': 'you', 'imo': 'in my opinion',
         'ily': 'i love you',
         'bf': 'boyfriend', 'gf': 'girlfriend', '5g': '5th generation', 'tldr': 'too long didn\'t read',
         'rofl': 'rolling on the floor laughing',
         'lmk': 'let me know', 'hmu': 'hit me up', 'tba': 'to be announced', 'asap': 'as soon as possible',
         'roi': 'return on investment',
         'tgif': 'thank goodness it\'s friday'}  ## Need a huge dictionary

    s = ' '.join(d[word] if word in d else word for word in s.split())


    # removing hashtags
    s = ' '.join([word.replace('#', '') for word in s.split()])

    # removes links
    s = ' '.join([re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", word) for word in s.split()])

    # reduce excess letters ie finallllly -> finally
    pattern = re.compile(r"(.)\1{2,}")
    s = ' '.join([pattern.sub(r"\1\1", word) for word in s.split()])

    # autocorrect (must run code that preceeds this function above)
    s = ''.join([word.term for word in sym_spell.lookup_compound(s, max_edit_distance=2)])

    # removing single letters except single digits or i or a (perform autocorrect first)
    s = ' '.join([w for w in s.split() if (len(w) > 1 or w.isdigit() or w.lower() == 'a' or w.lower() == 'i')])

    return s


sentdf['clean_tweet'] = sentdf['tweet'].apply(lambda x: preproc(x))
sentdf['nouns'] = ''
sentdf['polarity'] = np.nan
sentdf['subjectivity'] = np.nan
sentdf['compound'] = np.nan

noun_phrases = []

for i in range(len(sentdf)):
    # too sensitive
    tweet = sentdf['clean_tweet'][i]

    vader_opinion = sentiment_analyzer_scores(tweet)

    opinion = TextBlob(tweet)

    #print(opinion)

    sentdf.at[i, 'polarity'] = opinion.sentiment.polarity
    sentdf.at[i, 'subjectivity'] = opinion.sentiment.subjectivity
    # print(vader_opinion['compound'])
    sentdf.at[i, 'compound'] = vader_opinion['compound']
    sentdf.at[i, 'nouns'] = str(opinion.noun_phrases)
    #print(opinion.sentiment.subjectivity)
    #print(sentdf.at[i, 'subjectivity'])

    # print(opinion.noun_phrases)
    for noun in opinion.noun_phrases:
        noun_phrases.append(noun.lower().split(' '))
    # print(opinion.noun_phrases)



noun_phrases = list(chain.from_iterable(noun_phrases))
#noun_phrases = [s = [s for s in x if len(s) == 2]]

noun_phrases = [s for s in noun_phrases if len(s) > 1]

sentdf.to_pickle('sentdf.pkl')

# word frequencies
counts = collections.Counter(noun_phrases)
trend_words = pd.DataFrame(counts.most_common(30),
                            columns=['words', 'count'])

with open('../data/trends.pkl', 'wb') as f:
    pickle.dump(trend_words, f)
