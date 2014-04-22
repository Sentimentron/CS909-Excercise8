#!/usr/bin/env python

from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from collections import Counter
import re

def preprocess_tables():
    stopwords = nltk.corpus.stopwords.words('english')
    for train, topic, title, text in filtered_corpus():
        buf = []
        for line in text.split('\n'):
            if re.search('[^ ]+(  ).+(  )',line) is None:
                continue
            buf.append(line)
        table_str = ' '.join(buf)
        table_str = re.sub('[^a-zA-Z ]+', ' ', table_str)
        table_str = table_str.lower()
        yield train, topic, table_str.split(' ')

def export_to_arff(mode, output_path):
    assert mode in ["TRAIN", "TEST"]

    # Get table items
    corpus = [(topic, table) for train, topic, table in preprocess_tables() if train == mode]

    # Create some ARFF atributes
    attributes = set([])
    for topic, table in corpus:
        for word in table:
            word = word.strip()
            regexp = re.compile(r'[^a-zA-Z]')
            word = re.sub(regexp, '', word)
            if len(word) == 0:
                continue
            attributes.add(word)

    topics = set([])
    for topic, table in corpus:
        topics.add(topic)

    attributes = sorted(attributes)
    output_f = open(output_path, 'w')
    print >> output_f, '@relation "titles"'
    for f in attributes:
        print >> output_f, '@attribute %s {present, notPresent}' % (f,)
    print >> output_f, '@attribute topicClass {%s}' % (','.join(topics),)

    print >> output_f, '@data'
    for topic, title in corpus:
        row = dict(((u, 'notPresent') for u in attributes))
        for word in title:
            row[word] = 'present'
        buf = []
        for attr in attributes:
            buf.append(row[attr])
        buf.append(topic)
        print >> output_f, ','.join(buf)

    output_f.close()

if __name__ == "__main__":
    export_to_arff("TRAIN", "table_train.arff")
    export_to_arff("TEST", "table_test.arff")
