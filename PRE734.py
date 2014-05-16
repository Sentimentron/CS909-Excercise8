from collections import Counter
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, Ward
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, RandomizedPCA, TruncatedSVD
import numpy as np
from scipy import sparse
from time import time

def load_data():
    y = np.load("features.npy.npz")
    labels   = np.load("labels.npy")
    features = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape']).tocsc()
    topics   = np.load("topics.npy")
    return features, labels

if __name__ == "__main__":
    features, labels = load_data()
    distinct_labels = set(labels)
    print features.shape
