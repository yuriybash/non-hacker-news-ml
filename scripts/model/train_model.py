#!/usr/bin/env python
"""

Train a model with the given configuration and save it to the 'models' dir.
Currently, we hard-code the model and model configuration based off what is
found to be most effective (whether by accuracy, F-score, etc - that's
decided elsewhere).

TODO: Read a model config file to load, train, and save a model.

Usage:
    train_model [options]
    train_model -h | --help

Options:
    -h --help           Show this screen.
    --save-features     Whether to save the feature vectorizers
    --save-model        Whether to save the model
"""


from os.path import dirname, join

import docopt
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def generate_model():
    with open(join(dirname(dirname(__file__)), '../data/data.csv')) as f:
        data_df = pd.read_csv(f)

    title_vectorizer = TfidfVectorizer(ngram_range=[1, 1], max_features=500)
    title_vectorizer.fit(data_df.title)
    X_title = title_vectorizer.transform(data_df.title).toarray()

    url_vectorizer = TfidfVectorizer(ngram_range=[1,1], max_features=500)
    url_vectorizer.fit(data_df.url)
    X_url = url_vectorizer.transform(data_df.url).toarray()

    X = np.concatenate([X_title, X_url], axis=1)
    Y = data_df['noneng'].values

    clf = MultinomialNB(alpha=20.0)
    clf.fit(X, Y)

    if args['--save-model']:
        # TODO: use joblib instead of pickle once bug is fixed
        with open('title_vectorizer.pkl', 'w') as t_out_file:
            pickle.dump(title_vectorizer, t_out_file)

    if args['--save-features']:
        with open('url_vectorizer.pkl', 'w') as u_out_file:
            pickle.dump(url_vectorizer, u_out_file)

        with open('NB_model.pkl', 'w') as out_file:
            pickle.dump(clf, out_file)

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    generate_model()
