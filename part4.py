import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
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

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data, linkage_type, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = AgglomerativeClustering(linkage=linkage_type, n_clusters=n_clusters)
    model.fit(data_scaled)
    return model.labels_

def fit_modified(data, linkage_method, elbow_threshold=0.05):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    Z = linkage(data_scaled, method=linkage_method)
    distances = Z[:, 2]
    distance_diffs = np.diff(distances)
    normalized_distance_diffs = distance_diffs / distances.max()
    elbow_index = np.where(normalized_distance_diffs > elbow_threshold)[0]
    cutoff_distance = distances[elbow_index[0]] if len(elbow_index) > 0 else distances[-1]
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff_distance, linkage=linkage_method)
    model.fit(data_scaled)
    return model.labels_


def compute():
    answers = {}
    n_samples = 100
    random_state = 42
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    b = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)

    datasets_dict = {'nc': nc, 'nm': nm, 'bvv': bvv, 'add': add, 'b': b}
    answers["4A: datasets"] = datasets_dict
    linkage_methods = ['single', 'complete', 'ward', 'average']

    hierarchical_results = {}
    for dataset_name, (X, y) in datasets_dict.items():
        hierarchical_results[dataset_name] = {}
        for linkage_method in linkage_methods:
            labels = fit_hierarchical_cluster(X, linkage_method, 2)
            hierarchical_results[dataset_name][linkage_method] = labels

    plot_dct = {}
    for dataset_name in hierarchical_results:
        X, y = datasets_dict[dataset_name]
        plot_dct[dataset_name] = ((X, y), hierarchical_results[dataset_name])

    myplt.plot_part1C(plot_dct, "part4_b.jpg")

    modified_results = {}
    for dataset_name, (X, y) in datasets_dict.items():
        modified_results[dataset_name] = {}
        for linkage_method in linkage_methods:
            labels = fit_modified(X, linkage_method)
            modified_results[dataset_name][linkage_method] = labels

    plot_dct_modified = {}
    for dataset_name in modified_results:
        X, y = datasets_dict[dataset_name]
        plot_dct_modified[dataset_name] = ((X, y), modified_results[dataset_name])

    myplt.plot_part1C(plot_dct_modified, "part4_c.jpg")

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
