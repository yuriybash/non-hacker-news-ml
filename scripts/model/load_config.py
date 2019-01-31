from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from scripts.model.custom_exceptions import VectorizerLoadException, EstimatorLoadException

VECTORIZERS = {
        'count': CountVectorizer,
        'tfidf': TfidfVectorizer,
    }
ESTIMATORS = {
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'LinearSVC': LinearSVC,
        'MultinomialNB': MultinomialNB,
        'SVC': SVC,
        'RandomForestClassifier': RandomForestClassifier,
        'GaussianNB': GaussianNB,
        'Perceptron': Perceptron,
        'SGDClassifier': SGDClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'LogisticRegression': LogisticRegression,
        'MLPClassifier': MLPClassifier
    }


def get_vectorizer_cls(name):
    try:
        v_cls = VECTORIZERS[name]
        return v_cls
    except KeyError:
        raise VectorizerLoadException("Invalid vectorizer name: %s" % name)


def get_estimator_cls(name):
    try:
        e_cls = ESTIMATORS[name]
        return e_cls
    except KeyError:
        raise EstimatorLoadException("Invalid estimator name: %s" % name)
