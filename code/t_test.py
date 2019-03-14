######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# March of 2019
# t-test
######################################################################################

import json
import numpy as np 
import sys
import math

from random import shuffle
from bayes_function import bayes


trainingSetPath = "../datasets/tic-tac-toe.json"
testSetPath = "../datasets/tic-tac-toe.json"

# Load data
with open(trainingSetPath) as f:
    trainSet = json.load(f)
# Load test set
with open(testSetPath) as f:
    testSet = json.load(f)
Data = trainSet["data"]
foldSize = round((len(Data))/10)

# randomly shuffle data before separating folds
shuffle(Data)

# Print file header
print("TAN NaiveBayes")

# create fold and apply classifier
for i in range(10):
	testData = []
	trainingData = []
	testFoldIdx = [foldSize*i, foldSize*(i+1)]
	if testFoldIdx[1] > len(Data): testFoldIdx[1] = len(Data)
	for j in range(len(Data)):
		if j in range(testFoldIdx[0], testFoldIdx[1]+1):
			testData.append(Data[j])
		else:
			trainingData.append(Data[j])

	# TAN
	results_t = bayes(trainingSetPath, testSetPath, "t", trainingData, testData)
	predictions = [probability >= 0.5 for probability in results_t[1]]
	accuracy_t = sum([predictions[i] == results_t[0][i] \
		for i in range(len(predictions))])/len(predictions)

	# Naive Bayes
	results_n = bayes(trainingSetPath, testSetPath, "n", trainingData, testData)
	predictions = [probability >= 0.5 for probability in results_n[1]]
	accuracy_n = sum([predictions[i] == results_n[0][i] \
		for i in range(len(predictions))])/len(predictions)

	# print accuracy results
	print("%.4f"%accuracy_t, end = " ")
	print("&", end = " ")
	print("%.4f"%accuracy_n, end = " ")
	print("&", end = " ")
	print("%.4f"%(accuracy_t - accuracy_n), end = "\\\\ \n")















