from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, Ward
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, RandomizedPCA, TruncatedSVD
import numpy as np
from scipy import sparse
from time import time

def load_data():
    features = np.load("features.npy")
    labels   = np.load("labels.npy")
    return features, labels

def bench(estimator, X, name):
    # Lifted from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
    t0 = time()
    estimator.fit(X)
    print('% 9s   %.2fs   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_)
            ))

if __name__ == "__main__":
    features, labels = load_data()
    distinct_labels = set(labels)
    n_clusters = len(distinct_labels)
    data = sparse.csc_matrix(features)
    tsvd = TruncatedSVD(n_components=100)
    X = tsvd.fit_transform(data, labels)


    bench(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10), X, "KMEANS-TSVD")

    bench(Ward(n_clusters=n_clusters), X, "WARD")

    #print "Running PCA"

    #pca = RandomizedPCA(n_components = n_clusters).fit(data)
    #estimator = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
    #bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
    #          name="PCA-based",
    #          data=data)
