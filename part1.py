import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs, make_circles, make_moons
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
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_scaled)

    return kmeans.labels_

def compute():
    answers = {}
    n_samples = 100
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42)
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    b = datasets.make_blobs(n_samples=100, random_state=42)
    random_state = 42
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=42)

    datasets_list = [
        (nc, 'nc'),
        (nm, 'nm'),
        (b, 'b'),
        (add, 'add'),
        (bvv, 'bvv')
    ]
    dct = answers["1A: datasets"] = {name : data for data, name in datasets_list}

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    k_values = [2, 3, 5, 10]
    kmeans_results = {}
    for data, name in datasets_list:
        results = {}
        for k in k_values:
            labels = fit_kmeans(data[0], k)
            results[k] = labels
        kmeans_results[name] = (data, results)
    myplt.plot_part1C(kmeans_results, "plot1_C.jpg")
    dct = answers["1C: cluster successes"] = {"b": [3], "bvv": [3]} 
    dct = answers["1C: cluster failures"] = ["nc", "nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
    dct = answers["1D: datasets sensitive to initialization"] = ["nc", "nm", "add"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)