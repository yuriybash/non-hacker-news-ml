# non-hacker-news

This repo holds all of the code related to training, testing, and comparing ML models and pipelines for [Non-Hacker News](https://chrome.google.com/webstore/developer/edit/hpngeobpeckngjhdchikmijnkhfmedph), a project I've been working to make the non-technical content on [Hacker News](news.ycombinator.com) accessible to everyone.

For more information on how the end product works, please see [this](https://github.com/yuriybash/non-hacker-news-chrome) repo, which contains all of the extension-related code, as well as a general overview of the use case.

The `notebooks` directory has more technical information on how this works, but I'll provide a high-level overview in this README.


## The goal

The goal is ultimately to have a model that is able to differentiate between technical and non-technical posts on HN.

A few examples of technical posts:

- [What is the space overhead of Base64 encoding?](https://lemire.me/blog/2019/01/30/what-is-the-space-overhead-of-base64-encoding/)
- [Scripting in Common Lisp](https://ebzzry.io/en/script-lisp/)

A few examples of nontechnical posts:

- [Economic Analysis of Medicare-for-All](https://news.ycombinator.com/item?id=18613722)
- [The 9.9 Percent Is the New American Aristocracy](https://news.ycombinator.com/item?id=17172546)

A few examples of grey area posts (nontechnical posts on technical subjects or entities):

- [FaceTime bug lets you hear audio of person you are calling before they pick up](https://news.ycombinator.com/item?id=19022353)
- [Uber Fires More Than 20 Employees in Harassment Probe](https://news.ycombinator.com/item?id=14499294)

The goal is for whichever model gets chosen is to ultimately be able to predict whether a given story ((title, url) pair) is technical or nontechnical in nature.

## The data

Hacker News kindly provided 14.5m items available for download [here](https://archive.org/details/14566367HackerNewsCommentsAndStoriesArchivedByGreyPanthersHacker). It is a combination of both stories and other types of items (comments, etc).

Various data wrangling and cleaning scripts are in `scripts/data`.

We use the stories' `title` and `url` fields as features (and, ultimately, a [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectorizer). Each story is labeled 0 (technical) or 1 (nontechnical). The data ultimately looks like this:

![sample_data](https://github.com/yuriybash/non-hacker-news-ml/blob/master/assets/sample_data.png "sample_data")

## Testing different models

[scikit-learn](https://scikit-learn.org/stable/index.html)'s excellent [GridsearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) tool allows you to easily test various combinations of estimators and hyperparameters. You can check [grid_config.yml](https://github.com/yuriybash/non-hacker-news-ml/blob/master/grid_config.yml) for some of the configurations tested.

Ultimately, though GridsearchCV lets you quickly _set up_ estimators and hyperparameter combinations to test, training still takes a long time, and training time increases nonlinearly as the number of hyperparam/estimator combos go up.

Thus, the estimators and hyperparameters tested are a selection of likely effective classifiers. More on that in the notebooks.

## Evaluating different models

What metric should we be looking at? Consider the following options:

- high recall, low precision: few false positives - users sometimes don't see some nontechnical articles, but they rarely see any technical ones
- high precision, low recall: few false negatives - users sometimes see all nontechnical articles, but they also sometimes see technical ones

From the user's perspective, the latter sounds better - hence I prioritized precision as the key metric. Nonetheless, I've included accuracy below.

### Accuracy

Most models tested did fairly well with accuracy. Here is a comparison:

 - SVC (linear): 91%
 - Multinomial NB: 91%
 - Logistic Regression: 90%
 - Perceptron: 88%

(hyperparameters for these can be found in `grid_config.yml`)



### Precision

- SVC (linear): 0.83
- Multinomial NB: 0.94
- Logistic Regression: 0.84
- Perceptron: 0.85




### Caveats

Importantly, there is a _large_ grey area



## The model chosen

Ultimately, the model that currently runs the extension is a [multinomial naive Bayesian](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) model that returns probabilities of finding a given story ((title, URL) pair) to be nontechnical. You can find the configuration for this model [here](https://github.com/yuriybash/non-hacker-news-ml/blob/master/grid_config.yml).

It performed as well as the best models tested in accuracy, but more importantly - it had very good [precision](https://en.wikipedia.org/wiki/Precision_(information_retrieval)) score (as well as a high [F-score](https://en.wikipedia.org/wiki/F1_score)).

### Basic Overview

The grossly oversimplified intuition behind the naive Bayesian classifier is roughly:

1. First, determine the probability that an HN story ((title, URL) pair) is nontechnical given the presence of a each word (e.g. "politics" or "compiler") in the vocabulary
2. Second, determine the probability that an HN story is nontechnical given _all_ the words contained in it

To do this, we first compute the prior probabilities of each class - that is, what proportion of the total is represented by each class?

In this case, we only have two classes, 0 (technical) and 1 (nontechnical). There are 1626 nontechnical posts and 8263 technical posts - so ~0.164 and ~0.836, respectively.

We then compute the posterior probability for each word. That is, for each word, what is the probability of it occuring, _given_ a certain class?

For example, what is the posterior probability P("python" | 0) - what is the posterior probability for the word "python" for the technical class? How many times does the word "python" occur in (prelabeled) technical posts?

One way this can be calculated is by dividing the number of times "python" occurs in technical posts by the number of total words in the 0 class, _plus_ the total number of words in the training set (for smoothing).

There is one caveat: rather than use the simple count, we use a normalized [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) representation, in order to weigh each word based on how often it occurs in the text. See link for more details.

We then use these prior and posterior probabilities to compute the overall probability of a post being 0/1 - see [here](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering#Combining_individual_probabilities) for more information on how this is done.

### Results
