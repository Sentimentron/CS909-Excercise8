from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from collections import Counter
import re
import csv

def preprocess_docs():
    stopwords = nltk.corpus.stopwords.words('english')
    brown_tagged_sents = nltk.corpus.brown.tagged_sents(categories='news')
    unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
    raw_corpus = list(filtered_corpus())
    counter = 0
    for train, topic, title, text in raw_corpus:
        counter += 1
        if counter % 100 == 0:
            print "%.2f" % (counter * 100.0 / len(raw_corpus))
        text = nltk.word_tokenize(text) 
        text = unigram_tagger.tag(text)
        text = [(u, v) for u, v in text if u.lower() not in stopwords]
        text = [(u, v) for u, v in text if len(u) > 0]
        text = [(u, v) for u, v in text if v is not None]
        text = [(u, v) for u, v in text if len(v) > 0]
        text = [(u, v) for u, v in text if re.search("[^a-zA-Z]", v) is None]
        text = ["%s|%s" % (u, v) for u, v in text]
        yield train, topic, text

def export_to_arff(mode, output_path):
    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed titles
    corpus = [(topic, text) for train, topic, text in preprocess_docs() if train == mode]

    # Create some ARFF atributes
    attributes = set([])
    for topic, title in corpus:
        for word in title:
            attributes.add(word)

    topics = set([])
    for topic, title in corpus:
        topics.add(topic)

    attributes = sorted(attributes)
    output_f = open(output_path, 'w')
    print >> output_f, '@relation "docs"'
    for f in attributes:
        print >> output_f, '@attribute "%s" {present, notPresent}' % (f,)
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
    export_to_arff("TRAIN", "PRE710_train.arff")
    export_to_arff("TEST", "PRE710_test.arff")
