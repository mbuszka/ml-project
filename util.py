
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, Birch, DBSCAN
from itertools import chain


def read_files(size="small"):
    if size == "small":
        ratings = pd.read_csv("data/ml-latest-small/ratings.csv", sep=",", header=0, engine='python')
        movies = pd.read_csv("data/ml-latest-small/movies.csv", sep=",", header=0, engine='python')

        # Getting all the available movie genres
        genres = set(chain(*[x.split("|") for x in movies["genres"]]))
        genres.remove("(no genres listed)")

        # "genres vector"
        for v in genres:
            movies[v] = movies['genres'].str.contains(v)

        # movies mean raiting
        movies = movies.merge(ratings.groupby('movieId')['rating'].agg([pd.np.mean]), how='left', on='movieId')
        return ratings, movies, genres
