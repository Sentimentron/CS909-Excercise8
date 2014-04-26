#!/usr/bin/env python
"""
    This test is mostly adapted from 
        http://www.nltk.org/book/ch05.html
"""

import nltk
from nltk.corpus import brown

# Load the sentences
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

# Baseline: what's the most common tag?
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
most_common_tag = nltk.FreqDist(tags).max()
default_tagger = nltk.DefaultTagger(most_common_tag)
print "Default: %.2f" % (default_tagger.evaluate(brown_tagged_sents)*100.0,)

# Regexp
patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                     # nouns (default)
]
regexp_tagger = nltk.RegexpTagger(patterns)
print "Regexp: %.2f" % (regexp_tagger.evaluate(brown_tagged_sents)*100,)

# Lookup 
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.keys()[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print "Lookup: %.2f" % (baseline_tagger.evaluate(brown_tagged_sents) * 100,)

#N-grams - separate training and test data, 90% - 10%
size = len(brown_tagged_sents) * 0.9
size = int(size)

train_sents = brown_tagged_sents[:size]
test_sents  = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print "Unigram: %.2f" % (100 * unigram_tagger.evaluate(test_sents),)

bigram_tagger = nltk.BigramTagger(train_sents)
print "Bigram: %.2f" % (100 * bigram_tagger.evaluate(test_sents),)

# Ensemble
unigram_tagger = nltk.BigramTagger(train_sents, backoff=default_tagger)
bigram_tagger = nltk.BigramTagger(train_sents, backoff=unigram_tagger)
print "Ensemble: %.2f" % (100 * bigram_tagger.evaluate(test_sents),)


