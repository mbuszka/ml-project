#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as sa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from util import read_files
from argparse import ArgumentParser


def load_data(ratings, test_size, normalize, min_ratings):
    train, test = train_test_split(ratings, test_size=test_size)
    # unique, indices = np.unique(train['movieId'].values, return_inverse=True)
    
    counts = train.groupby('movieId')['rating'].count()
    counts = counts[counts >= min_ratings]
    movies = set(counts.index)
    mapping = pd.Series(data=range(len(counts.index)), index=counts.index)

    train = train[train.movieId.isin(movies)].copy()
    train['movieId'] = train.movieId.map(mapping)
    
    users = set(train['userId'])
    
    test = test[test.movieId.isin(movies) & test.userId.isin(users)].copy()
    test['movieId'] = test.movieId.map(mapping)
    
    if normalize == 'movieId' or normalize == 'userId':
        eps = 0.001
        means = train.groupby(normalize)['rating'].mean() - eps
        train['rating'] = train['rating'] - means.loc[train[normalize].array].values
        test['normalization_delta'] = means.loc[test[normalize].array].values
    else:
        test['normalization_delta'] = 0
    
    return train, test


def test_svd(ratings, size, min_ratings):
    err_svd = []
    cases = [(it, norm) for it in range(3) for norm in ['userId', 'movieId']]
    if size == 'big':
        ks = [20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    else:
        ks = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40]

    for it, normalize in tqdm(cases, desc='SVD', position=0):
        train, test = load_data(ratings, test_size=10000, normalize=normalize, min_ratings=min_ratings)
        utility = coo_matrix(
            (train['rating'], (train['userId'], train['movieId']))
            ).tocsr()
        for k in tqdm(ks, desc='Inner', position=1, leave=False):
            u, s, vt = sa.svds(utility, k=k)
            u = u @ np.diag(s)
            rmse = 0
            mae = 0
            num_samples = len(test.index)
            for r in test.itertuples():
                p = r.normalization_delta + u[r.userId, :] @ vt[:, r.movieId]
                rmse += np.float64((p - r.rating) ** 2)
                mae += np.float64(abs(p - r.rating))
            rmse /= num_samples
            rmse = np.sqrt(rmse)
            mae /= num_samples
            err_svd.append({ 'k': k, 'rmse': rmse, 'mae': mae, 'it': it, 'normalize': normalize })
    f = f"svd-{size}-{min_ratings}.csv"
    pd.DataFrame(err_svd).to_csv(f)


def calculate_item_item_similarity(ratings):
    """
    Calculate matrix of cosine distances between movies
    treated as vectors of ratings 
    """
    # Calculate the dot-products of ratings for all movies
    # Use sparse matrix for calculation to accomodate large data-set (~260k users, ~53k movies)
    # Storing results as dense matrix, allows us to use in-place operations
    # and provides faster indexing
    ii_similarity = coo_matrix(
        (ratings['rating'], (ratings['movieId'], ratings['userId']))
        ).tocsr()
    ii_similarity = ii_similarity.dot(ii_similarity.transpose()).toarray()
    
    d = 1 / np.sqrt(ii_similarity.diagonal())
    ii_similarity *= d.reshape((-1, 1))
    ii_similarity *= d.reshape((1, -1))
    return ii_similarity


def test_cf(ratings, size, min_ratings):
    err_ii = []

    cases = [(it, norm) for it in range(3) for norm in ['none', 'movieId', 'userId']]

    for _, normalize in tqdm(cases, desc='CF', position=0):
        train, test = load_data(ratings, test_size=10000, normalize=normalize, min_ratings=min_ratings)
        ii_similarity = calculate_item_item_similarity(train)
        user_ratings = train.groupby('userId').indices
        rmse = 0
        mae = 0
        no_match = 0
        num_samples = len(test.index)
        for r in test.itertuples():
            other = train.iloc[user_ratings[r.userId]]
            similarities = ii_similarity[r.movieId, other.movieId]
            p = np.sum(similarities * other['rating'])
            d = similarities.sum()
            if d == 0:
                no_match += 1
            else:
                p = p/d + r.normalization_delta
                rmse += np.float64((p - r.rating) ** 2)
                mae += np.float64(abs(p - r.rating))
        rmse /= num_samples
        rmse = np.sqrt(rmse)
        mae /= num_samples
        err_ii.append({ 'normalize': normalize, 'rmse': rmse, 'mae': mae, 'no_match': no_match })
    
    f = f"cf-{size}-{min_ratings}.csv"
    pd.DataFrame(err_ii).to_csv(f)


def test_knn(ratings, size, min_ratings):
    err_knn = []
    cases = [(it, norm) for it in range(3) for norm in ['none', 'movieId', 'userId']]
    ks = [5, 6, 7, 8, 9, 10, 15, 20, 30]

    for it, normalize in tqdm(cases, desc='KNN', position=0):
        train, test = load_data(ratings, test_size=10000, normalize=normalize, min_ratings=min_ratings)
        ii_similarity = calculate_item_item_similarity(train)
        user_ratings = train.groupby('userId').indices
        for k in tqdm(ks, desc='Inner', position=1, leave=False):
            s = 0
            num_samples = len(test.index)
            for r in test.itertuples():
                other = train.iloc[user_ratings[r.userId]].copy()
                other['sim'] = ii_similarity[r.movieId, other.movieId]
                l = min(k, len(other.index))
                p = other.sort_values(by='sim', ascending=False).head(l)['rating'].mean()
                p = r.normalization_delta + p
                s += np.float64((p - r.rating) ** 2)
            s /= num_samples
            s = np.sqrt(s)
            err_knn.append({ 'k': k, 'err': s, 'it': it, 'normalize': normalize })

    f = f"knn-{size}-{min_ratings}.csv"
    pd.DataFrame(err_knn).to_csv(f)

if __name__ == "__main__":
    parser = ArgumentParser(description='Run tests')
    parser.add_argument('--size', dest='size', default='small')
    args = parser.parse_args()
    ratings = read_files(size=args.size)[0]

    # test_svd(ratings, args.size, args.min_ratings)
    # test_cf(ratings, args.size, args.min_ratings)
    test_svd(ratings, args.size, 1)
    test_svd(ratings, args.size, 2)
    test_svd(ratings, args.size, 5)
    test_svd(ratings, args.size, 20)