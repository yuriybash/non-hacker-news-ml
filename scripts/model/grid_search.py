import datetime
import pprint
from os.path import dirname, join

import pandas as pd
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from load_config import get_vectorizer_cls, get_estimator_cls
from transformers import ItemSelector


OUTFILE = ('../../results/results_%s' % str(datetime.datetime.utcnow()))


def parse_data(data_path):
    with open(join(dirname(dirname(__file__)), data_path)) as f:
        data_df = pd.read_csv(f)
    return data_df


def train_test_models():
    with open(join(dirname(dirname(__file__)), 'grid_config.yml')) as f:
        config = yaml.safe_load(f)

    data_df = parse_data(config['data'])
    train_models(config['models'], data_df, config)


def train_models(models, data, config):

    with open(OUTFILE, 'a') as out_f:
        out_f.write("STARTING TRAINING AT %s\n" % str(datetime.datetime.utcnow()))

    for id_, model_cfg in models.items():
        try:
            train_model(model_cfg, data, config['cross_validation'], config['test'])
            print("Done training model ID # %s at %s" % (id_, datetime.datetime.utcnow()))
        except Exception as e:
            print "Error encountered for model %s" % id_
            continue

        print "\nTRAINING FINISHED AT %s\n" % str(datetime.datetime.utcnow())


def train_model(model_cfg, data_df, cv_cfg, test_cfg):

    estimator_cfg = model_cfg['estimator']

    title_cfg = model_cfg['vectorizer']['title']
    url_cfg = model_cfg['vectorizer']['url']

    t_vectorizer = get_vectorizer_cls(title_cfg['name'])()
    u_vectorizer = get_vectorizer_cls(url_cfg['name'])()
    estimator = get_estimator_cls(estimator_cfg['name'])()

    pipeline, parameters = create_pipeline_params(
        t_vectorizer, u_vectorizer, estimator, title_cfg, url_cfg, estimator_cfg)

    X_train, X_test, Y_train, Y_test = train_test_split(
        data_df[['title', 'url']],
        data_df['noneng'],
        test_size=float(cv_cfg['train_test_split'][-1])/100, random_state=42
    )

    print("---------------------------------------------------------------------------")
    for score in test_cfg['scores']:
        print("# Tuning hyper-parameters for a %s model for %s\n" %
              (estimator.__class__, score))

        clf = GridSearchCV(
            pipeline,
            parameters,
            cv=cv_cfg['n_folds'],
            scoring='%s_macro' % score if score!='accuracy' else score,
            n_jobs=-1
        )

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        record_results(pprint.pprint, score, estimator, clf, X_test, Y_test, Y_pred)
        out_f = open("%s_%s_%s" % (score, OUTFILE, datetime.datetime.utcnow()), 'w')
        record_results(out_f.write, score, estimator, clf, Y_test, Y_pred)
        out_f.close()


def record_results(record_func, score, estimator, clf, Y_test, y_pred):
    record_func("----------------------------------------")
    record_func(
        "\nBest parameters set found on development set for class %s, "
        "score '%s':\n" %
        (estimator.__class__, score)
    )
    record_func(str(clf.best_params_))
    record_func("\n\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        record_func("%0.3f (+/-%0.03f) for %r\n"
                    % (mean, std * 2, params))
    record_func("\nDetailed classification report:\n")
    record_func("The model is trained on the full development set.\n")
    record_func("The scores are computed on the full evaluation set.\n")
    record_func(classification_report(Y_test, y_pred))


def create_pipeline_params(t_vect, u_vect, estimator, t_cfg, u_cfg, e_cfg):

    pipeline = Pipeline([

        ('union', FeatureUnion(
            transformer_list=[

                ('title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('vec', t_vect),
                ])),

                ('url', Pipeline([
                    ('selector', ItemSelector(key='url')),
                    ('vec', u_vect),
                ])),

            ],

        )),

        ('estimator', estimator),
    ])

    parameters = []

    for t_param_group in t_cfg['parameters']:
        for u_param_group in u_cfg['parameters']:
            for e_param_group in e_cfg['parameters']:

                combined_param_group = {}

                for t_key, t_val in t_param_group.iteritems():
                    combined_param_group['__'.join(['union', 'title', 'vec', t_key])] = t_val

                for u_key, u_val in u_param_group.iteritems():
                    combined_param_group['__'.join(['union', 'url', 'vec', u_key])] = u_val

                for e_key, e_val in e_param_group.iteritems():
                    combined_param_group['__'.join(['estimator', e_key])] = e_val

                parameters.append(combined_param_group)

    return pipeline, parameters


if __name__ == '__main__':
    train_test_models()
