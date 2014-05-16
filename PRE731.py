from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, Ward
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, RandomizedPCA, TruncatedSVD
import numpy as np
from scipy import sparse
from time import time
from collections import Counter

def load_data():
    y = np.load("features.npy.npz")
    labels   = np.load("labels.npy")
    features = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape']).tocsc()
    topics   = np.load("topics.npy")
    return features, labels

def get_most_likely_class_map(estimated_labels, actual_labels):
    counter = {}
    for i, j in zip(estimated_labels, actual_labels):
        if i not in counter:
            counter[i] = Counter()
        counter[i].update([j])
    ret = {}
    for i in counter:
        c, _ = counter[i].most_common(1)[0]
        ret[i] = c
    return ret

def print_classification_metrics(estimated_labels, actual_labels):
    mapping = get_most_likely_class_map(estimated_labels, actual_labels)
    predicted_labels = []
    for i in estimated_labels:
        predicted_labels.append(mapping[i])
    
    print metrics.classification_report(estimated_labels, actual_labels)

def bench(estimator, X, name, labels):
    # Lifted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
    t0 = time()
    estimator.fit(X)
    print_classification_metrics(estimator.labels_, labels)
    print('% 9s   %.2fs   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(X, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=300)))

if __name__ == "__main__":
    features, labels = load_data()
    distinct_labels = set(labels)
    n_clusters = len(distinct_labels)
    data = sparse.csc_matrix(features)
    tsvd = TruncatedSVD(n_components=10)
    X = tsvd.fit_transform(data, labels)

    bench(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10), X, "KMEANS-TSVD", labels)

    bench(Ward(n_clusters=n_clusters), X, "WARD", labels)

    #print "Running PCA"

    #pca = RandomizedPCA(n_components = n_clusters).fit(data)
    #estimator = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
    #bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
    #          name="PCA-based",
    #          data=data)

