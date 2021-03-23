"""
This code is just for predicting new instances and preproc is to
be passed as an arguement to the tdif vectorizer for new test instances.

"""

import unidecode
import re
import pkg_resources
from symspellpy import SymSpell, Verbosity
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


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

#Stemming
ps = PorterStemmer()

#Lemmitization
lemmatizer = WordNetLemmatizer()

def preproc(s):
    """
    s is the string/tweet
    manipulate s and return it
    Tfidfvectorizer's preprocessor should .apply this function to every tweet/string
    """

    # removing numbers this isn't working (this will leave extra spaces if digit is enclosed by two spaces)
    s = ''.join([i for i in s if not i.isdigit()])

    # Removing all the greek characters using unidecode library
    s = ' '.join([unidecode.unidecode(word) for word in s.split()])

    # removing mentions @
    s = ' '.join([word for word in s.split() if not word.startswith("@")])

    # Removing the word 'hmm' and it's variants
    s = ' '.join([word for word in s.split() if not word == 'h(m)+'])

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

    # start of new preprocessing stuff

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

    # remove stop words
    s = ' '.join([word for word in s.split() if not word in set(stopwords.words('english'))])

    # lemmatization
    s = ' '.join([lemmatizer.lemmatize(word) for word in s.split()])

    # stemming
    s = ' '.join([ps.stem(word) for word in s.split()])

    return s