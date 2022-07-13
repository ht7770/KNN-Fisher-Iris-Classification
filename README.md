# KNN-Fisher-Iris-Classification
Implementation of K-nearest neighbours classifier on the fisher iris dataset.

K-nearest Neighbours is a classification model that determines the class of a data point by the classes of the data points around it, the K refers to the number of neighbours that the classification is based on, 
for example if K is 5 then the model will choose the 5 closest data points if there is a majority in a certain class then the new data point will be given that class label as well. 

To test this model I divided the included fisheriris dataset into 5 blocks, with 4 blocks training the model and the last block testing the model. I then used k-fold cross validation on these blocks to provide a 
less bias estimate of the model skill than a simple train/test split. 
