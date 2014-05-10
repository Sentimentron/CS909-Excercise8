from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from collections import Counter
import re
import csv
import sys

from nltk.corpus import wordnet as wn

import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

import pdb
def preprocess_docs():
    stopwords = nltk.corpus.stopwords.words('english')
    corpus = list(filtered_corpus())
    counter = 0
    for train, topic, title, text in corpus:
        if topic not in ["wheat", "grain"]:
            continue
        if counter % 10 == 0:
            print "%.2f %%\r" % (counter * 100.0 / len(corpus),),
            sys.stdout.flush()
        counter += 1
        text = [i for i in nltk.word_tokenize(title) if i.lower() not in stopwords]
        buf = []
        for word in text:
            synsets = wn.synsets(word)
            grain = []
            wheat = []
            for s in synsets:
                grain.append(s.path_similarity(wn.synset('wheat.n.02')))
                wheat.append(s.path_similarity(wn.synset('grain.n.08')))

            grain = [i for i in grain if i is not None]
            wheat = [i for i in wheat if i is not None]

            if len(grain) == 0:
                grain = 0
            else:
                grain = sum(grain) * 1.0 / len(grain)
            if len(wheat) == 0:
                wheat = 0
            else:
                wheat = sum(wheat) * 1.0 / len(wheat)
            buf.append((word, grain, wheat))
        yield train, topic, buf
    print ""

def get_features(mode, attributes=None, topics=None):
    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed titles
    corpus = [(topic, text) for train, topic, text in preprocess_docs() if train == mode]

    # Create attributes
    if attributes is None:
        attributes = set([])
        for topic, title in corpus:
            for word, grain, wheat in title:
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

    # Construct float matrix
    X = np.zeros((len(corpus), len(attributes)*2), dtype='float')
    Y = [0 for _ in range(len(corpus))]

    if topics is None:
        topics = set([])
        for topic, _ in corpus:
            topics.add(topic)

        # Construct a topic mapping
        topics = dict([(j, i) for i, j in enumerate(topics)])

    for row, (topic, title) in enumerate(corpus):
        if topic not in topics:
            pdb.set_trace()
            continue
        for word, grain, wheat in title:
            if word not in attributes:
                continue
            offset = attributes[word]
            X[row][offset] = grain
            X[row][len(attributes)+offset] = wheat
        Y[row] = topics[topic]

    return X, Y, attributes, topics

if __name__ == "__main__":
    print "Reading..."
    Xtrain, Ytrain, attributes, topics = get_features("TRAIN")
    Xtest, Ytest, _, _ = get_features("TEST", attributes, topics)
    for clf in [LinearSVC, SVC, MultinomialNB]:
        clf = clf()
        print clf
        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)
        print metrics.accuracy_score(Ytest, Ypred)
        print metrics.confusion_matrix(Ytest, Ypred)
        print metrics.classification_report(Ytest, Ypred, topics.values(), topics.keys())
