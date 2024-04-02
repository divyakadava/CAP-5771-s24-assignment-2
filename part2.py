from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_scaled)
    sse = np.sum((data_scaled - kmeans.cluster_centers_[kmeans.labels_])**2)
    return kmeans.labels_, sse



def compute():
    # ---------------------
    answers = {}
    X, y = make_blobs(n_samples=20, centers=5, center_box=(-20, 20), random_state=12)

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [X, y]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    sse_values = []
    for k in range(1, 9):
        _, sse = fit_kmeans(X, k)
        sse_values.append(sse)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 9), sse_values, marker='o')
    plt.title('SSE as a function of k')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.savefig("sse_plot.png")
    plt.close()

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = sse_values

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    inertia_values = []
    for k in range(1, 9):
        kmeans = KMeans(n_clusters=k, init='random', random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)

    # Plotting inertia
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 9), inertia_values, marker='o')
    plt.title('Inertia as a function of k')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.savefig("inertia_plot.png")
    plt.close()

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = inertia_values
    sse_diff = np.diff(sse_values)
    inertia_diff = np.diff(inertia_values)
    optimal_k_sse = np.argmin(sse_diff) + 1  # +1 because index starts at 0
    optimal_k_inertia = np.argmin(inertia_diff) + 1

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes" if optimal_k_sse == optimal_k_inertia else "no"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
