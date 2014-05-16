from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from nltk.corpus import brown
from collections import Counter
import re
import csv

def preprocess_docs():
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.PorterStemmer()
    brown_tagged_sents = brown.tagged_sents(categories='news')
    fd = nltk.FreqDist(brown.words(categories='news'))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    likely_tags = dict((word, cfd[word].max()) for word in fd.keys())
    tagger= nltk.UnigramTagger(model=likely_tags)

    for train, topic, title, text in filtered_corpus():
        tagged = []
        for s in nltk.sent_tokenize(text):
            for i, j in tagger.tag(nltk.word_tokenize(s)):
                if j != None:
                    tagged.append((i, j))
                else:
                    tagged.append((i, "NA"))
        yield train, topic, ["%s/%s" % (i, j) for i, j in tagged]

def export_to_arff(mode, output_path):
    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed titles
    corpus = [(topic, text) for train, topic, text in preprocess_docs() if train == mode]

    # Create some ARFF atributes
    attributes = set([])
    for topic, title in corpus:
        for word in title:
            print word
            regexp = re.compile(r'[^/a-zA-Z]')
            if regexp.search(word) is not None:
                continue
            attributes.add(word)

    topics = set([])
    for topic, title in corpus:
        topics.add(topic)

    attributes = sorted(attributes)
    output_f = open(output_path, 'w')
    print >> output_f, '@relation "docs"'
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
    export_to_arff("TRAIN", "PRE795_train.arff")
    export_to_arff("TEST", "PRE795_test.arff")
