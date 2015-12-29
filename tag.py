import re, json, pprint
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import *
from sklearn.datasets import fetch_20newsgroups

pp = pprint.PrettyPrinter(indent=2)

with open('./noaa.json', 'rb') as f:
    noaa = json.load(f)

# categories = [
#     'alt.atheism',
#     'talk.religion.misc',
# ]
# # Uncomment the following to do the analysis on all the categories
# #categories = None

# print("Loading 20 newsgroups dataset for categories:")
# print(categories)

# data = fetch_20newsgroups(subset='train', categories=categories)
# print("%d documents" % len(data.filenames))
# print("%d categories" % len(data.target_names))
# print()

# print data

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    # ('clf', SGDClassifier()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    data = []
    for entry in tqdm(noaa):
        # data.extend(re.split('\s', entry[u'description']))
        # for chain in entry[u'keyword']:
        #     data.extend(re.split('>', chain))
        data.append(entry[u'description'])
        data.extend(entry[u'keyword'])

    X = pipeline.fit_transform(data)
    print X.shape
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    # grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pp.pprint(parameters)
    # t0 = time()
    # grid_search.fit(data.data, data.target)
    # print("done in %0.3fs" % (time() - t0))
    # print()

    # print("Best score: %0.3f" % grid_search.best_score_)
    # print("Best parameters set:")
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))