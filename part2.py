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
    # SSE calculation is updated as per the task requirement.
    sse = np.sum((data_scaled - kmeans.cluster_centers_[kmeans.labels_])**2)
    return kmeans.labels_, sse



def compute():
    # ---------------------
    answers = {}
    X, y = make_blobs(n_samples=20, centers=5, center_box=(-20, 20), random_state=12)
    # We need to return the coordinates X, the labels y, and the center points, which we get from the KMeans f

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
    k_values = list(range(1, 9))  # k values from 1 to 8
    for k in k_values:
        _, sse = fit_kmeans(X, k)
        sse_values.append(sse)  # Appending only the SSE value

    # Plotting SSE values
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sse_values, marker='o')  # Plot using k_values and sse_values
    plt.title('SSE as a function of k')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.xticks(k_values)  # Ensure x-axis ticks correspond to k_values
    plt.savefig("sse_plot.png")
    plt.close()

    # Saving SSE values
    dct = answers["2C: SSE plot"] = list(zip(k_values, sse_values)) 

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    inertia_values = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='random', random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)  # Appending only the inertia value

    # Plotting inertia values
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_values, marker='o')  # Plot using k_values and inertia_values
    plt.title('Inertia as a function of k')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.xticks(k_values)  # Ensure x-axis ticks correspond to k_values
    plt.savefig("inertia_plot.png")
    plt.close()

    # Saving inertia values
    dct = answers["2D: inertia plot"] = list(zip(k_values, inertia_values))

    changes_sse = np.diff(sse_values)  # Compute the difference between each SSE value
    changes_inertia = np.diff(inertia_values)  # Compute the difference for inertia

    # Now we look for the "elbow" in the differences, which is the point where the change is minimal
    optimal_k_sse = np.argmin(changes_sse) + 2  # +2 since np.diff reduces the length by 1, and we need the index for k
    optimal_k_inertia = np.argmin(changes_inertia) + 2

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes" if optimal_k_sse == optimal_k_inertia else "no"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
