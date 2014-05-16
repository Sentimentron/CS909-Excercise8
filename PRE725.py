from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, RandomizedPCA
import numpy as np
from scipy import sparse
from time import time

def load_data():
    y = np.load("features.npy.npz")
    print y
    labels   = np.load("labels.npy")
    features = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape']).tocsc()
    return features, labels

def bench_k_means(estimator, name, data, labels):
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

if __name__ == "__main__":
    data, labels = load_data()
    distinct_labels = set(labels)
    n_clusters = len(distinct_labels)

    bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
              name="k-means++", data=data, labels=labels)
    bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
               name="random", data=data, labels=labels)

    #print "Running PCA"

    pca = RandomizedPCA(n_components = n_clusters).fit(data)
    estimator = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
              name="PCA-based",
              data=data, labels=labels)

