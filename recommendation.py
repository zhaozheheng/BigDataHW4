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
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

if __name__== "__main__":
    # create spark session
    conf = SparkConf().setMaster("local").setAppName("hw4_recommendation")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    # load data and convert it to spark dataframe
    lines = spark.read.text('ratings.dat').rdd
    parts = lines.map(lambda row: row.value.split('::'))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    ratings.show()
    print('ratings loaded!')
    (training, test) = ratings.randomSplit([0.7, 0.3])

    als = ALS(rank=10, maxIter=20, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    print("Mean-square error = " + str(rmse * rmse))
    sc.stop()