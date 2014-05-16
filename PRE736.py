from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, RandomizedPCA
import numpy as np
from scipy import sparse
from time import time
from collections import Counter
import itertools
import pydbscan
import random

def load_data():
    y = np.load("features.npy.npz")
    labels   = np.load("labels.npy")
    features = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape']).tocsc()
    topics   = np.load("topics.npy")
    return features, labels

def get_most_likely_class_map(estimated_labels, actual_labels):
    counter = {}
    for i, j in zip(estimated_labels, actual_labels):
        if i == 0: 
            continue
        if i not in counter:
            counter[i] = Counter()
        counter[i].update([j])
    ret = {}
    for i in counter:
        c, _ = counter[i].most_common(1)[0]
        ret[i] = c
    return ret

def print_classification_metrics(estimated_labels_, actual_labels_):

    estimated_labels = []
    actual_labels = []
    for i, j in zip(estimated_labels_, actual_labels_):
        estimated_labels.append(i)
        actual_labels.append(j)
    mapping = get_most_likely_class_map(estimated_labels, actual_labels)
    predicted_labels = []
    for i in estimated_labels:
        if i == 0:
            predicted_labels.append(-1)
            continue
        predicted_labels.append(mapping[i])

    return metrics.classification_report(predicted_labels, actual_labels), predicted_labels, actual_labels

class DBSCANSparse(object):

    def __init__(self, eps, minpts):
        self.eps = eps
        self.minpts = minpts
        self.clusters = None
        self.label_mapping = {}
        self.labels_ = []

    @classmethod
    def nextpow2(cls, of):
        i = 1
        while i < of:
            i *= 2
        return i

    def fit(self, data):
        X, y = data
        documents, labels = X.shape
        print "%d labels" % (labels,)
        print "%d documents" % (documents,)

        quadtree_size = max(self.nextpow2(documents), self.nextpow2(labels))
        print "%d quadtree" % (quadtree_size,)

        quadtree = pydbscan.create_quadtree(quadtree_size, quadtree_size)

        print "building quadtree..."
        for i in np.transpose(np.nonzero(X)):
            if pydbscan.quadtree_insert(quadtree, int(i[0]), int(i[1])) == 0:
                print "warning", i

        print
        print "clustering..."
        self.labels_ = pydbscan.pyDBSCAN(quadtree, documents, self.eps, self.minpts)

def bench(estimator, name, data):
    # Lifted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
    t0 = time()
    features, labels = data
    estimator.fit(data)
    report, pred, actual = print_classification_metrics(estimator.labels_, labels)
    print report
    print max(estimator.labels_)
    c = Counter(estimator.labels_)
    total = 0
    for k in c:
        total += c[k]
    print 1.0 * total / max(estimator.labels_)
        
    print '% 9s   %.2fs   %.3f   %.3f   %.3f   %.3f   %.3f' % (name, (time() - t0),
             metrics.homogeneity_score(actual, estimator.labels_),
             metrics.completeness_score(actual, estimator.labels_),
             metrics.v_measure_score(actual, estimator.labels_),
             metrics.adjusted_rand_score(actual, estimator.labels_),
             metrics.adjusted_mutual_info_score(actual, estimator.labels_))

    fp.close()

if __name__ == "__main__":
    features, labels = load_data()
    eps = 0.89
    minpts = 2
    d = DBSCANSparse(eps, minpts)
    bench(d, "DBSCAN_%.2f_%d" % (eps, minpts), (features, labels))
