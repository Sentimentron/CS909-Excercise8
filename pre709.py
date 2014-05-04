from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from collections import Counter
import re
import csv

def preprocess_docs():
    stopwords = nltk.corpus.stopwords.words('english')
    for train, topic, title, text in filtered_corpus():
        text = [i for i in nltk.word_tokenize(text) if i not in stopwords]
        yield train, topic, text

def export_to_arff(mode, output_path):
    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed titles
    corpus = [(topic, text) for train, topic, text in preprocess_docs() if train == mode]

    # Create some ARFF atributes
    attributes = set([])
    for topic, text in corpus:
        for word in text:
            regexp = re.compile(r'[^a-zA-Z]')
            if regexp.search(word) is not None:
                continue
            attributes.add(word)

    topics = set([])
    for topic, text in corpus:
        topics.add(topic)

    attributes = sorted(attributes)
    output_f = open(output_path, 'w')
    print >> output_f, '@relation "docs"'
    for f in attributes:
        print >> output_f, '@attribute %s numeric' % (f,)
    print >> output_f, '@attribute topicClass {%s}' % (','.join(topics),)

    print >> output_f, '@data'
    for topic, text in corpus:
        row = dict(((u, 0) for u in attributes))
        for word in text:
            if word not in row:
                continue
            row[word] += 1
        buf = []
        for attr in attributes:
            buf.append(row[attr])
        buf.append(topic)
        print >> output_f, ','.join(map(str,buf))
    output_f.close()

if __name__ == "__main__":
    export_to_arff("TRAIN", "PRE709_train.arff")
    export_to_arff("TEST", "PRE709_test.arff")
