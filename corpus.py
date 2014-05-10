#!/usr/bin/env python

import nltk
from lxml import etree
from collections import Counter

def read_corpus():
    with open('converted.xml','r') as fp:
        root = etree.parse(fp)
        for reuters in root.findall("Reuters"):
            for topic in reuters.findall(".//Topic"):
                title = reuters.find("Title")
                text  = reuters.find("Text")
                if title is None:
                    continue
                if text is None:
                    continue
                yield (reuters.get("lewissplit"), topic.text, title.text, text.text)

def filtered_corpus():
    for train, topic, title, text in read_corpus():
        if topic not in ["earn","acq","money-fx","crude","grain","trade","interest","wheat","ship","corn"]:
            continue
        yield train, topic, title, text

def pos_tag_corpus():
    for train, topic, title, text in filtered_corpus():
        title = nltk.word_tokenize(title)
        tagged_text = nltk.word_tokenize(text)
        tagged_text = nltk.pos_tag(tagged_text)
        yield train, topic, title, tagged_text

def most_common_categories():
    c = Counter([])
    for train, topic, title, text in read_corpus():
        c.update([topic])
    return c.most_common(10)

if __name__ == "__main__":
    for t in pos_tag_corpus():
        print t
