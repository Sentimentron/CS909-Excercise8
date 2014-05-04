from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, RandomizedPCA
import numpy as np
from scipy import sparse
from time import time
import itertools

import pydbscan

def load_data():
    features = np.load("features.npy")
    labels   = np.load("labels.npy")
    return features, labels

def bench(estimator, name, data):
    # Lifted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=300)))

def nextpow2(of):
    i = 1
    while i < of:
        i *= 2
    return i

if __name__ == "__main__":
    features, labels = load_data()
    distinct_labels = set(labels)
    n_clusters = len(distinct_labels)

    documents, labels = features.shape
    print "%d labels" % (labels,)
    print "%d documents" % (documents,)

    quadtree_size = max(nextpow2(documents), nextpow2(labels))
    print "%d quadtree" % (quadtree_size,)

    quadtree = pydbscan.create_quadtree(quadtree_size, quadtree_size)

    print "building quadtree..."
    for i in np.transpose(np.nonzero(features)):
        if pydbscan.quadtree_insert(quadtree, int(i[0]), int(i[1])) == 0:
            print "warning", i

    print
    print "clustering..."
    clusters = pydbscan.pyDBSCAN(quadtree, documents, 0.9, 10)
    print clusters

    np.save('clusters.npy', clusters)
