"""
abuse model is where we take an unprocessed tweet and preprocess it and transform it into
a TFIDF matrix so that it can be inputted into the logistic regression model that we trained
in jupyter notebooks.
THis function is defined in predict() and is to be imported to wherever it is needed.
"""
import pickle
from notebook_abuse_model.preproc import preproc
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# load the logistic regression model
# the below will cause as error is this file's curwd is notebook_abuse_model
try:
    with open('notebook_abuse_model\log_reg_model.pkl', 'rb') as f:
        log_reg = pickle.load(f)
    # load feature dataframe
    df2 = pd.read_pickle('notebook_abuse_model/data/tfidf_feature_matrix.pkl')
except FileNotFoundError:
    with open('log_reg_model.pkl', 'rb') as f:
        log_reg = pickle.load(f)
    # load feature dataframe
    df2 = pd.read_pickle('data/tfidf_feature_matrix.pkl')


def predict(s, threshold=0.5):
    sample_input = s
    sample_input = preproc(sample_input)
    sample_input = [sample_input]

    vectorizer2 = TfidfVectorizer()
    Xtest = vectorizer2.fit_transform(sample_input)
    denseTest = Xtest.todense()
    denseTestlist = denseTest.tolist()
    featurenamesTest = vectorizer2.get_feature_names()

    df_test = pd.DataFrame(denseTestlist, columns=featurenamesTest)

    # stackoverflow answer below
    not_existing_cols = [c for c in df2.columns.tolist() if c not in df_test]
    df_test = df_test.reindex(df_test.columns.tolist() + not_existing_cols, axis=1)
    df_test.fillna(0, inplace=True)
    df_test = df_test[df2.columns.tolist()]  # use the original X structure as mask for the new inference dataframe
    df_test = df_test.drop(['labelxyz'], axis=1)

    # yhat = log_reg.predict(df_test)
    a = log_reg.predict_proba(df_test)
    if a[0][0] > threshold:
        return 'no'
    else:
        return 'yes'

#print(type(predict("I love you")))