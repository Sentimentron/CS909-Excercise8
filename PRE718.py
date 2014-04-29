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
    print "Creating input... (titles)"
    Xtrain, Ytrain, attributes, topics = get_features("TRAIN", "TITLE")
    Xtest, Ytest, _, _ = get_features("TEST", "TITLE", attributes, topics)
    clf = MultinomialNB()
    clf.fit(Xtrain, Ytrain)
    Ytitlepred = clf.predict_proba(Xtest)

    print "Creating input... (text)"
    Xtrain, Ytrain, attributes, topics = get_features("TRAIN", "TEXT")
    Xtest, Ytest, _, _ = get_features("TEST", "TEXT", attributes, topics)
    clf = LinearSVC()
    clf.fit(Xtrain, Ytrain)
    Ytextpred = clf.predict(Xtest)
    Ytextscores = clf.decision_function(Xtest)
    Ytextmaxscore = np.max(np.abs(Ytextscores))
    Ytextscores = np.abs(Ytextscores) / Ytextmaxscore

    Y = [0 for _ in Ytextpred]
    classifierF1Text = {
        "earn": 0.91,
        "wheat": 0.10,
        "money-fx": 0.61,
        "corn": 0.09,
        "trade": 0.73,
        "acq": 0.90,
        "grain": 0.29,
        "interest": 0.45,
        "crude": 0.72,
        "ship": 0.53
    }
    classifierF1Title = {
        "earn": 0.96,
        "wheat": 0.04,
        "money-fx": 0.70,
        "corn": 0.02,
        "trade": 0.72,
        "acq": 0.90,
        "grain": 0.46,
        "interest": 0.50,
        "crude": 0.71,
        "ship": 0.39
    }
    inverse_topic_map = dict([(topics[k], k) for k in topics])
    for row, (y1, y2) in enumerate(zip(Ytitlepred, Ytextscores)):
        distribution = [0.0 for _ in topics]
        for col, y in enumerate(y1):
            distribution[col]  = y * classifierF1Title[inverse_topic_map[col]]
        for col, y in enumerate(y2):
            distribution[col] += y * classifierF1Text[inverse_topic_map[col]]
        # Find the maximuum likelihood
        y = 0
        prob = 0
        for col, score in enumerate(distribution):
            if score > prob:
                y = col
                prob = score
        Y[row] = y

    print metrics.accuracy_score(Ytest, Y)
    print metrics.confusion_matrix(Ytest, Y)
    print metrics.classification_report(Ytest, Y, topics.values(), topics.keys())