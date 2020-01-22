import os
import subprocess
from itertools import chain
from pathlib import Path

import pandas as pd
import sklearn


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


def export_tree(tree, feature_names, filename):
    Path("output/").mkdir(parents=True, exist_ok=True)
    with open("output/tmp1.dot", "w") as f:
        sklearn.tree.export_graphviz(tree, out_file=f, feature_names=feature_names)

    cmd = ['dot', '-Tpdf', 'output/tmp1.dot', '-o', "output/" + filename]
    subprocess.call(cmd)
    os.remove("output/tmp1.dot")
