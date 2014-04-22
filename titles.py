#!/usr/bin/env python

from corpus import filtered_corpus
from gensim import corpora, models, similarities
import nltk
from collections import Counter
import re

def preprocess_titles():
    stopwords = nltk.corpus.stopwords.words('english')
    for train, topic, title, text in filtered_corpus():
        title = title.upper()
        title = [i for i in nltk.word_tokenize(title) if i not in stopwords]
        yield train, topic, title

def export_to_arff(mode, output_path):
    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed titles
    corpus = [(topic, title) for train, topic, title in preprocess_titles() if train == mode]

    # Create some ARFF atributes
    attributes = set([])
    for topic, title in corpus:
        for word in title:
            regexp = re.compile(r'[^a-zA-Z]')
            if regexp.search(word) is not None:
                continue
            attributes.add(word)

    topics = set([])
    for topic, title in corpus:
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

def build_title_model(mode, dictionary=None):
    # Parameter check
    assert mode in ["TRAIN", "TEST"]

    # Read preprocessed titles
    corpus = [(topic, title) for train, topic, title in preprocess_titles() if train == mode]

    # Remove words which only appear once
    word_counter = Counter([])
    for topic, title in corpus:
        word_counter.update(title)

    thresholded = []
    for topic, title in corpus:
        reconstructed = []
        for word in title:
            if word_counter[word] < 2:
                continue
            reconstructed.append(word)
        thresholded.append((topic, reconstructed))

    # Convert words into identifiers
    corpus_rep = [title for topic, title in thresholded]
    counter = 0
    if dictionary is None:
        dictionary = corpora.Dictionary(
            corpus_rep
        )
    print dictionary
    ret = [dictionary.doc2bow(title) for topic, title in thresholded]
    topics = [topic for topic, title in thresholded]
    return topics, ret, dictionary

def build_lsi_representation(corpus_tuple):
    topics, corpus, dictionary = corpus_tuple
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10)
    corpus_lsi = lsi[corpus_tfidf]
    return topics, tfidf, lsi, corpus_lsi, dictionary

# To evaluate:
# Build a classifier, feed original topics, plus LSI weights to build representation
# Convert evaluation set into the right format

def print_arff_lsi_header(output_f, topics, categories):
    print >> output_f, '@relation "titles"'
    print >> output_f, '@attribute word string' % (f,)
    for f in range(topics):
        print >> output_f, '@attribute topic_%d numeric' % (f,)
    print >> output_f, '@attribute category {%s}' % (','.join(categories))
    print >> output_f,"@data"

def print_arff_lsi_data(output_f, topics, corpus_lsi):
    for topic, doc in zip(topics, corpus_lsi):
        for i,val in doc:
            print >> output_f, val, ',',
        print >> output_f, topic

def evaluate(model_tuple1, corpus_tuple2):
    _, term_model, semantic_model, _, dictionary = model_tuple1
    topics, corpus, dictionary2 = corpus_tuple2
    corpus_term = term_model[corpus]
    sematic_model = semantic_model[corpus]

if __name__ == "__main__":
    #export_to_arff("TRAIN", "title_train.arff")
    #export_to_arff("TEST", "title_test.arff")

    corpus_tuple = build_title_model("TRAIN")
    topics, tfidf, lsi, corpus_lsi, dictionary = build_lsi_representation(corpus_tuple)
    output_f = open('title_train_lsi.arff', 'w')
    print_arff_lsi_header(output_f, 10, set(topics))
    print_arff_lsi_data(output_f, topics, corpus_lsi)
    output_f.close()

    corpus_tuple = build_title_model("TEST", corpus_tuple[2])
    topics, tfidf, lsi, corpus_lsi, dictionary = build_lsi_representation(corpus_tuple)
    output_f = open('title_test_lsi.arff', 'w')
    print_arff_lsi_header(output_f, 10, set(topics))
    print_arff_lsi_data(output_f, topics, corpus_lsi)
    output_f.close()

