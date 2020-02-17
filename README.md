# BigDataHW4

Homework 4 consists of three parts. The first one focuses on clustering handwritten questions, where you have to clearly show the steps and results asked in each item. Second part is still about clustering, but now a programming question to be implemented using Spark. Finally, in part 3 you have a programming question related to recommendation systems.
Note: Please find the files required for programming questions attached in eLearning.
PART 1 – CLUSTERING (Handwritten questions)
(Q1) Consider the following eight points in 2-dimensional space:
(2,10); (2,5); (8,4); (5,9); (7,5); (6,4); (1,2); (4,9); (10,10).
Suppose we plan to use the Euclidean distance metric and that we are interested in clustering these points into 3 clusters.
(a) Plot the data to see what might be appropriate clusters.
(b) Beginning with the points (2,5), (5,8) and (4,9) as initial cluster centers,
form the three initial clusters.
(c) Use the k-means clustering algorithm to get the final three clusters. What
are the resulting centers and resulting clusters? (Here K = 3).
(d) Re-run steps (a), (b) and (c) with starting points (10,5), (2,9) and (3,3).
(Q2) Use the similarity matrix in Table below to perform single and complete link hierarchical clustering. Show your results by drawing a dendrogram.
Note: The dendrogram should clearly show the order in which the points are merged.
 
 (Q3) Hierarchical clustering is sometimes used to generate K clusters, K > 1 by taking the clusters at the Kth level of the dendrogram (root is at level 1.)
By looking at the clusters produced in this way, we can evaluate the behavior of hierarchical clustering on different types of data and clusters, and also compare hierarchical approaches to K-means.
The following is a set of one-dimensional points: {6, 12, 18, 24, 25, 28, 30, 42, 48}.
(a) For each of the following sets of initial centroids, create two clusters by assigning each point to the nearest centroid, and then calculate the total squared error for each set of two clusters.
Show both the clusters and the total squared error for each set of centroids.
1){5,7.5} 2) {15, 25} 3) {1, 50} 4) {19, 20}
(b) Do all of the centroids represent stable solutions, i.e., if the K-means algorithm was run on this set of points using the given centroids as the starting centroids, would there be any change in the clusters generated?
(c) What are the clusters produced by MIN? (MIN is single-link clustering, also called minimum method)
PART 2 – CLUSTERING (Programming question)
(Q4) Using spark machine learning library spark-mlib, use kmeans to cluster the movies using the ratings given by the user, that is, use the item-user matrix from itemusermat File provided as input to your program.
The itemusermat file contains the ratings given to each movie by the users in Matrix format. The file contains the ratings by users for 1000 movies.
Each line contains the movies id and the list of ratings given by the users. A rating of

0 is used for entries where the user did not rate a movie.
Below, we show an example of the format of Itemusermat file with the item-user matrix. Note that here, user 1 did not rate movie 2, so we use a rating of 0.
       user1 user2
          movie1 4 3
          movies2 0 2
      Set the number of clusters k = 10.
Your Scala/python code should produce the following output: For each cluster, print any 5 movies in the cluster. Your output should contain the movie_id, movie title, genre and the corresponding cluster it belongs to.
Note: Use the movies.dat file to obtain the movie title and genre.
Example of output format:
cluster: 1
123,Star wars, sci-fi
PART 3 – RECOMMENDATION SYSTEM (Programming question)
(Q5) Use Collaborative filtering to find the accuracy of ALS model. For this question, you will use ratings.dat file, which contains: User id :: movie id :: ratings :: timestamp.
Your program should report the accuracy of the model.
For details follow the link: https://spark.apache.org/docs/latest/mllib-collaborative- filtering.html
Please use 70% of the data for training and 30% for testing and report the MSE of the model.
