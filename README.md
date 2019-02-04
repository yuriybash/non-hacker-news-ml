# non-hacker-news

This repo holds all of the code related to training, testing, and comparing ML models and pipelines for [Non-Hacker News](https://chrome.google.com/webstore/detail/non-hacker-news/hpngeobpeckngjhdchikmijnkhfmedph), a project I've been working on to make the non-technical content on [Hacker News](news.ycombinator.com) accessible to everyone.

For more information on how the end product works, please see [this](https://github.com/yuriybash/non-hacker-news-chrome) repo, which contains all of the [Chrome extension](https://chrome.google.com/webstore/detail/non-hacker-news/hpngeobpeckngjhdchikmijnkhfmedph) related code, as well as a general overview of the use case.

The `notebooks` directory has more technical information the model selection and training process, but I'll provide a high-level overview in this README.


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

The goal is for our ultimate model to be able to predict whether a given story is technical or nontechnical in nature.

More specifically, given a story _X_, we want the model to predict that:

```
P(technical|X) > P(nontechnical|X)
```

for technical stories and

```
P(technical|X) < P(nontechnical|X)
```


## The data
### Raw data

Hacker News kindly provided 14.5m items available for download [here](https://archive.org/details/14566367HackerNewsCommentsAndStoriesArchivedByGreyPanthersHacker). It is a combination of both stories and other types of items (comments, etc). We are concerned with filtering stories exclusively.

Each story, in its raw form, looks like:

```
{
  "body": {
    "title": "The Dutch Have Solutions to Rising Seas",
    "url": "https://www.nytimes.com/interactive/2017/06/15/world/europe/climate-change-rotterdam.html",
    "descendants": 0,
    "by": "vincentmarle",
    "score": 1,
    "time": 1497574399,
    "type": "story",
    "id": 14565773
  },
  "source": "firebase",
  "id": 14565773,
  "retrieved_at_ts": 1497564307
}
```

The relevant fields to us are `title` and `url`. Both fields are cleaned up prior to training, and we ultimately use the domain in place of `url`.

Thus, the problem can be reformulated as the search for a model that predicts:

```
P(technical|(title,domain) > P(nontechnical|(title,domain)
```

for technical stories

and

```
P(technical|(title,domain) < P(nontechnical|(title,domain)
```

for nontechnical stories.

### Cleaning up the data

The `scripts/data` directory contains various data wrangling scripts. After running some basic clean up on the raw data, we are left with an unlabeled data set that looks like:

![sample_data_unlabeled](https://github.com/yuriybash/non-hacker-news-ml/blob/master/assets/sample_data_unlabeled.png "sample_data_unlabeled")

### Labeling the data

How should we label the data? There's a few options available:

- Discrete/binary: every story is 0 or 1 - either technical or nontechnical
- Discrete/nonbinary: every story falls into one of N categories, where each category represents the level of technical detail in it (i.e. 1 == not technical, 5 == somewhat technical, 10 == very technical)
- Continuous: there are a few different ways this could work, but one idea is to give each story a "technical score", which can take on any value in a given range.

We can eliminate option #3 immediately - there is no advantage to allowing, for example, all floats in the range `[0, 10]`, rather than just the integers in it (option #2).

The decision between #1 and #2 is more difficult. It is true (as mentioned [earlier](#the-goal)) that there is a significant grey area of articles, in which each article is not fully nontechnical and not fully technical either. This would suggest choosing option #1. However, there is an alternative solution: use a binary scale during labeling and then use the probabilities returned by the classifier when actually doing the filtering. This offers two advantages:

- labeling will be much faster, since the labeler can round up or down for articles in the grey area
- we can choose our own threshold when filtering articles later on

There _is_ potentially some valuable information lost when training our model, because it would be better to feed the classifier a fine-grained technical score rather than 0/1, but the trade-off in labeling speed, combined with the use of probability predictions (rather than classification predictions) makes this trade-off seem like one worth taking. As it turns out, binary classification is enough to achieve fairly high accuracy rates.

After the labeling the data, we are left with 1626 nontechnical articles and 8263 technical articles (a somewhat skewed data set).

## Testing different models

We are left with a fairly standard classification problem (K=2) and a somewhat skewed data set. There are a few go-to models typically used for this type of problem:

- logistic regression
- naive bayesian
- support vector machines

as well as a few others.

Even after choosing a subset of likely effective models, given the combinatorial nature of model and hyperparameter testing creating a replicable pipeline is crucial.

[scikit-learn](https://scikit-learn.org/stable/index.html)'s excellent [GridsearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) is really useful for this, and allows you to quickly create and test different pipelines.

Being able to express vectorizer-model-hyperparameter combinations in config is even more useful, which is why I did exactly that - `grid_config.yml` stores various models and hyperparameter combinations, and `scripts/grid_search` tests and records the results. This allows you to quickly try different ideas just by changing config.

### Handling skew

Our data set is somewhat imbalanced (16.4% base case). Some models handle this better than others. It's also worth trying to account for this skew. One way to do this is oversampling from the minority class or undersampling the majority class. The [imbalanced-learn]() library is a very useful library for this. In my tests, the [SMOTE](https://www.jair.org/index.php/jair/article/view/10302/24590) technique was more effective. Results (on average across multiple classifiers):


|        | Accuracy | Precision | Recall |
|:------:|:--------:|:---------:|:------:|
|  Base  |     0    |     0     |    0   |
| Random |   +2.1%  |   +3.2%   |  +0.1% |
|  SMOTE |   +7.8%  |   +5.6%   |  +0.4% |
| ADASYN |   +5.3%  |   +3.5%   |    0   |

The training data was preprocessed using SMOTE. The technique is fairly intuitive and involves creating synthetic minority class examples by generating examples within the bounds of the minority class hyperplane bounds:

![smote](https://raw.githubusercontent.com/rikunert/SMOTE_visualisation/master/SMOTE_R_visualisation_3.png "smote")
[image credit](http://rikunert.com/SMOTE_explained)

Test data remained isolated prior to testing.

## Results

Accuracy scores only tell part of the story, particularly with this type of skewed data set. A model can predict all 0s and still get 80% accuracy.

[Precision](https://en.wikipedia.org/wiki/Precision_and_recall), which measures  and [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) are key here.

What these metrics mean here:

- high recall, low precision: few false positives - users sometimes don't see some nontechnical articles, but they rarely see any technical ones
- high precision, low recall: few false negatives - users sometimes see all nontechnical articles, but they also sometimes see technical ones

From the user's perspective, we'd probably prefer the latter over the former, so precision will be a key metric in evaluating results.

|                            | Accuracy | Precision | Recall | F-Score |
|:--------------------------:|:--------:|:---------:|:------:|:-------:|
|     SVM (linear kernel)    |   0.910  |   0.865   |  0.795 |   0.91  |
|             LR             |   0.900  |   0.920   |  0.778 |   0.91  |
| Multinomial Naive Bayesian |   0.905  |   0.920   |  0.795 |   0.89  |

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

#### Accuracy

Most models tested did fairly well with accuracy. Here is a comparison:

 - SVC (linear): 91%
 - Multinomial NB: 91%
 - Logistic Regression: 90%
 - Perceptron: 88%

(hyperparameters for these can be found in `grid_config.yml`)



#### Precision

- SVC (linear): 0.83
- Multinomial NB: 0.94
- Logistic Regression: 0.84
- Perceptron: 0.85

Because we prioritize precision for this use case, the Multinomial NB classifier was chosen.

### Deploying the model

The model was serialized and deploy to AWS Lambda. For more on this, see [this]() repo - it's fairly straightforward.

### Next steps

There are many improvements to be made, including:

- relabeling the data with a "technical" score, rather than 0 or 1
- labeling more data (currently, n=10k)
- add online learning - so that users can report mislabeled posts
- retrain with deep learning (starting with an [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network))
