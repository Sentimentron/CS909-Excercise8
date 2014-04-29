from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from collections import Counter
import re
import csv
import sys

from nltk.corpus import wordnet as wn

def preprocess_docs():
    stopwords = nltk.corpus.stopwords.words('english')
    corpus = list(filtered_corpus())
    counter = 0
    for train, topic, title, text in corpus:
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

def export_to_arff(mode, output_path):
    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed titles
    corpus = [(topic, text) for train, topic, text in preprocess_docs() if train == mode]

    # Create some ARFF atributes
    attributes = set([])
    for topic, title in corpus:
        for word, grain, wheat in title:
            if re.search('[^a-zA-Z]',word) is not None:
                continue
            attributes.add("%s_grain" % (word,))
            attributes.add("%s_wheat" % (word,))

    topics = set([])
    for topic, title in corpus:
        topics.add(topic)

    attributes = sorted(attributes)
    output_f = open(output_path, 'w')
    print >> output_f, '@relation "docs"'
    for f in attributes:
        print >> output_f, '@attribute %s numeric' % (f,)
    topics = ['grain','wheat','notGrainOrWheat']
    print >> output_f, '@attribute topicClass {%s}' % (','.join(topics),)

    print >> output_f, '@data'
    for topic, title in corpus:
        row = dict(((u, 0) for u in attributes))
        for word, grain, wheat in title:
            row["%s_grain" % (word,)] = grain
            row["%s_wheat" % (word,)] = wheat 
        buf = []
        for attr in attributes:
            buf.append("%.4f" % (row[attr],))
        if topic in ['grain','wheat']:
            buf.append(topic)
        else:
            buf.append('notGrainOrWheat')
        print >> output_f, ','.join(buf)
    output_f.close()

if __name__ == "__main__":
    export_to_arff("TRAIN", "PRE713_train.arff")
    export_to_arff("TEST", "PRE713_test.arff")
