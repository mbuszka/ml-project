#!/bin/bash


if [ $1 = "small" ]
then
    wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip -O data/data_small.zip
    cd data
    unzip -o data_small.zip 
    rm data_small.zip
    cd ..
fi

if [ $1 = "big" ]
then
    wget http://files.grouplens.org/datasets/movielens/ml-latest.zip -O data/data_big.zip
    cd data
    unzip -o data_big.zip 
    rm data_big.zip
    cd ..
fi