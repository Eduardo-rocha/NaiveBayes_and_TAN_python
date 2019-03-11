#!/usr/bin/env python3.6
# chmod +x fileName.py

######################################################################################
# CS 760, Spring 2019
# HW 2
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# March of 2019
# Problem 3: t-test
######################################################################################

import json
import numpy as np 
import sys
import math

from bayes_function import bayes



trainingSetPath = "../datasets/tic-tac-toe.json"
testSetPath = "../datasets/tic-tac-toe.json"

with open(trainingSetPath) as f:
    trainSet = json.load(f)

# Load test set
with open(testSetPath) as f:
    testSet = json.load(f)

Data = trainSet["data"]
foldSize = round((len(Data))/10)


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

	results_t = bayes(trainingSetPath, testSetPath, "t", trainingData, testData)
	print(results_t[0])
	predictions = [probability >= 0.5 for probability in results_t[1]]

	accuracy_t = sum([predictions[i] == results_t[0][i] \
		for i in range(len(predictions))])/len(predictions)



	print(accuracy_t)


	#accuracy_t.append([])

	results_n = bayes(trainingSetPath, testSetPath, "n", trainingData, testData)
















