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
            buf.append(line)
            print line
        table_str = ' '.join(buf)
        float_buf = []
        for i in table_str.split(' '):
            if i == 'billion':
                if len(float_buf) > 0:
                    float_buf[-1] *= 1.0E9
                continue
            if i == 'mln' or i == 'million':
                if len(float_buf) > 0:
                    float_buf[-1] *= 1.0E6
                continue
            cur = []
            for j in i:
                if j >= '0' and j <= '9':
                    cur.append(j)
                if j == '.':
                    if '.' in cur:
                        cur = []
                        break
                    cur.append(j)
            if len([c for c in cur if c != '.']) == 0:
                continue
            cur = float(''.join(cur))
            float_buf.append(cur)
        table_str = table_str.lower()
        yield train, topic, float_buf

def export_to_arff(mode, output_path):
    assert mode in ["TRAIN", "TEST"]

    # Get table items
    corpus = [(topic, table) for train, topic, table in preprocess_tables() if train == mode]

    # Create some ARFF atributes
    attributes = set(['value_prev','value_1'])
    topics = set([])
    for topic, table in corpus:
        topics.add(topic)

    attributes = sorted(attributes)
    output_f = open(output_path, 'w')
    print >> output_f, '@relation "tablevals"'
    for f in attributes:
        print >> output_f, '@attribute %s numeric' % (f,)
    print >> output_f, '@attribute topicClass {%s}' % (','.join(topics),)

    print >> output_f, '@data'
    for topic, title in corpus:
        value_prev = None
        for entry in title:
            if entry >= 1E12:
                continue
            if value_prev is None:
                print >> output_f, '?,',
            else:
                print >> output_f, value_prev, ',',
            print >> output_f, ','.join([str(entry), topic])
            value_prev = entry

    output_f.close()

if __name__ == "__main__":
    export_to_arff("TRAIN", "tablevals_train.arff")
    export_to_arff("TEST", "tablevals_test.arff")
