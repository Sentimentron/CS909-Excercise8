from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, RandomizedPCA
import numpy as np
from scipy import sparse
from time import time
import itertools

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

if __name__ == "__main__":
    features, labels = load_data()
    distinct_labels = set(labels)
    n_clusters = len(distinct_labels)

    distance_matrix = sparse.lil_matrix(len(features),len(features))

    for (id1, row1), (id2, row2) in itertools.product(enumerate(features), enumerate(features)):
        row = row1 * row2
        if True in row:
            distance_matrix[id1][id2] = 0

    db = DBSCAN(eps=0.3, min_samples=10, metric="precomputed").fit(distance_matrix)
    bench(db, "DBSCAN", features)

    #bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #          name="k-means++", data=features)
    #bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
    #            name="random", data=features)

    #print "Running PCA"

    #pca = RandomizedPCA(n_components = n_clusters).fit(data)
    #estimator = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
    #bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
    #          name="PCA-based",
    #          data=data)

