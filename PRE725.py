from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, RandomizedPCA
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
        counter[i].update([i])
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
    

def bench_k_means(estimator, name, data, labels):
    # Lifted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
    t0 = time()
    estimator.fit(data)
    print_classification_metrics(estimator.labels_, labels)
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

if __name__ == "__main__":
    data, labels = load_data()
    distinct_labels = set(labels)
    n_clusters = len(distinct_labels)
    bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
              name="k-means++", data=data, labels=labels)
    bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
               name="random", data=data, labels=labels)
    pca = RandomizedPCA(n_components = 10).fit(data)
    estimator = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
              name="PCA-based",
              data=data, labels=labels)

