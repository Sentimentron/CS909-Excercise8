from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, RandomizedPCA
import numpy as np
from scipy import sparse
from time import time
from collections import Counter
import itertools
import pydbscan

def load_data():
    features = np.load("features.npy")
    labels   = np.load("labels.npy")
    clusters = np.load("clusters.npy")
    return features, labels, clusters

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
        self.clusters = pydbscan.pyDBSCAN(quadtree, documents, self.eps, self.minpts)

        for l, c in zip(y, clusters):
            if c == 0:
                continue
            mapper_counter.update([(l, c)])

        self.label_mapping = {}
        for (l, c), p in mapper_counter.most_common():
            if c not in label_mapping:
                label_mapping[c] = l

        for c in clusters:
            if c not in label_mapping:
                label = -1
            else:
                label = label_mapping[c]
            self.labels_.append(label)


def bench(estimator, name, data):
    # Lifted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
    t0 = time()
    features, labels = data
    estimator.fit(data)
    fp = open('dbscan.tuning', 'a')
    print >> fp, '% 9s   %.2fs   %.3f   %.3f   %.3f   %.3f   %.3f' % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_))

if __name__ == "__main__":
    features, labels, clusters = load_data()

    mapper_counter = Counter([])

    for l, c in zip(labels, clusters):
        if c == 0:
            continue
        mapper_counter.update([(l, c)])

    labels = Counter(labels)
    clusters = Counter(clusters)

    print clusters.most_common()
    print labels.most_common()

    print mapper_counter.most_common()

    label_mapping = {}
    for (l, c), p in mapper_counter.most_common():
        if c not in label_mapping:
            label_mapping[c] = l

    print label_mapping
    features, labels, clusters = load_data()
    for eps, minpts in itertools.product([i/100.0 + 0.95 for i in range(5)], [100, 150, 200]):
        d = DBSCANSparse(eps, minpts)
        bench(d, "DBSCAN_%.2f_%d" % (eps, minpts), (features, labels))
