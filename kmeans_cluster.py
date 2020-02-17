# -*- coding: utf-8 -*-

# system libs
import string
import re
from time import sleep
import json, sys
import time
# external libs
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

    # lines = spark.read.text('ratings.dat').rdd
    # parts = lines.map(lambda row: row.value.split('::'))
    # ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
    # ratings = spark.createDataFrame(ratingsRDD)

if __name__== "__main__":
    # set cluster number
    clusterNum = 10
    # create spark session
    conf = SparkConf().setMaster("local").setAppName("hw4_kmeans")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    # load movie details and set it into movieId, title and genre
    lines = spark.read.text('movies.dat').rdd
    parts = lines.map(lambda row: row.value.split('::'))
    movie_infoRDD = parts.map(lambda p: Row(movieId=int(p[0]), title=str(p[1]),genre=str(p[2])))
    movie_info = spark.createDataFrame(movie_infoRDD)
    movie_info.show()
    print('movie details loaded')
    # load data and set it into movieId and ratings with vectors format which can be used in kmeans model
    lines = spark.read.text('itemusermat').rdd
    parts = lines.map(lambda row: row.value.split(' '))
    movie_ratingsRDD = parts.map(lambda p: Row(movieId=int(p[0]), ratings=Vectors.dense(list(map(float, p[1:])))))
    movie_ratings = spark.createDataFrame(movie_ratingsRDD)
    movie_ratings.show()
    print('movie rating loaded')
    # print out data scale
    print("data size: " + str(movie_ratings.count()))
    # setup Kmean model
    kmeans = KMeans().setK(clusterNum).setFeaturesCol('ratings').setPredictionCol('prediction').fit(movie_ratings)
    # clustering data
    res = kmeans.transform(movie_ratings)
    # print out results for each cluster
    print('-----------------------------------------------------------------')
    for i in range(clusterNum):
        print('cluster: ' + str(i))
        # only get 5 results at most for each cluster
        cluster = res.filter(res.prediction == i).limit(5)
        # cluster.join(movie_info, 'movieId', 'inner').show()
        cluster = cluster.join(movie_info, 'movieId', 'inner').collect()
        for mov in cluster:
            print(str(mov[0]) + ',' + str(mov[4]) + ',' + str(mov[3]))
        print('-----------------------------------------------------------------')
        # cluster.limit(5).show()
    # res.show()

    sc.stop()