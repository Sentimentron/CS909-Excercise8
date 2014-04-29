from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from collections import Counter
import re
import csv

import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

import pdb

def preprocess_docs(feature):
    assert feature in ["TITLE", "TEXT"]
    stopwords = nltk.corpus.stopwords.words('english')
    for train, topic, title, text in filtered_corpus():
        if feature == "TITLE":
            text = title
        text = [i for i in nltk.word_tokenize(text) if i not in stopwords]
        yield train, topic, text

def get_features(mode, feature, attributes=None, topics=None):

    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed text
    corpus = [(topic, text) for train, topic, text in preprocess_docs(feature) if train == mode]

    # Create attributes
    if attributes is None:
        attributes = set([])
        for topic, title in corpus:
            for word in title:
                regexp = re.compile(r'[^a-zA-Z]')
                if regexp.search(word) is not None:
                    continue
                attributes.add(word)
        # Construct a columnar mapping
        attributes = sorted(attributes)
        attributes_dict = {}
        counter = 0
        for a in attributes:
            attributes_dict[a] = counter
            counter += 1
        attributes = attributes_dict

    # Construct binary matrix
    X = np.zeros((len(corpus), len(attributes)), dtype='bool')
    Y = [0 for _ in range(len(corpus))]

    if topics is None:
        topics = set([])
        for topic, _ in corpus:
            topics.add(topic)

        # Construct a topic mapping
        topics = dict([(j, i) for i, j in enumerate(topics)])

    #metrics.classification_report

    # Build the features matrix
    for row, (topic, title) in enumerate(corpus):
        if topic not in topics:
            pdb.set_trace()
            continue
        for word in title:
            if word not in attributes:
                continue
            offset = attributes[word]
            X[row][offset] = 1
        Y[row] = topics[topic]

    return X, Y, attributes, topics

if __name__ == "__main__":
    for clf, feature in zip([LinearSVC, MultinomialNB], ["TEXT", "TITLE"]):
        Xtrain, Ytrain, attributes, topics = get_features("TRAIN", feature)
        Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(
            Xtrain, Ytrain, test_size = 0.4
        )
        print "**classification_report**"
        print clf
        clf = clf()
        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)
        print metrics.accuracy_score(Ytest, Ypred)
        print metrics.confusion_matrix(Ytest, Ypred)
        print metrics.classification_report(Ytest, Ypred, topics.values(), topics.keys())
