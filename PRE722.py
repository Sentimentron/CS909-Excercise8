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

import pdb

def preprocess_docs():
    stopwords = nltk.corpus.stopwords.words('english')
    for train, topic, title, text in filtered_corpus():
        textbuf = []
        for sentence in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sentence):
                if word != '.':
                    textbuf.append(word)
        text = textbuf
        title = [i for i in nltk.word_tokenize(title) if i.lower() not in stopwords]
        yield train, topic, text + title

def get_features(mode, attributes=None, topics=None):

    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed text
    corpus = [(topic, text) for train, topic, text in preprocess_docs() if train == mode]

    # Create attributes
    if attributes is None:
        attributes = set([])

        attr_counter = Counter([])
        for topic, title in corpus:
            prev = None
            for word in title:
                attr_counter.update([word])
                if prev is not None:
                    bigram = "%s|%s" % (prev, word)
                    attr_counter.update([bigram])
                prev = word

        most_common = set([word for word, count in attr_counter.most_common(50)])

        for topic, title in corpus:
            prev = None
            for word in title:
                regexp = re.compile(r'[^a-zA-Z\.\-\/\']')
                if regexp.search(word) is not None:
                    print word
                    continue
                if attr_counter[word] < 3:
                    continue
                if word in most_common:
                    continue
                attributes.add(word)
                if prev is not None:
                    bigram = "%s|%s" % (prev, word)
                    if bigram in most_common:
                        continue
                    if attr_counter[word] < 3:
                        continue
                    attributes.add(bigram)
                prev = word

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
        prev = None
        for word in title:
            if word in attributes:
                offset = attributes[word]
                X[row][offset] = 1
            if prev is not None:
                bigram = "%s|%s" % (prev, word)
                if bigram in attributes:
                    offset = attributes[bigram]
                    X[row][offset] = 1
            prev = word

        Y[row] = topics[topic]

    return X, Y, attributes, topics

if __name__ == "__main__":
    print "Creating input..."
    Xtrain, Ytrain, attributes, topics = get_features("TRAIN")
    Xtest, Ytest, _, _ = get_features("TEST", attributes, topics)
    print "Testing..."
    for clf in [LinearSVC]:
        print "**classification_report**"
        print clf
        clf = clf()
        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)
        print metrics.accuracy_score(Ytest, Ypred)
        print metrics.confusion_matrix(Ytest, Ypred)
        print metrics.classification_report(Ytest, Ypred, topics.values(), topics.keys())
