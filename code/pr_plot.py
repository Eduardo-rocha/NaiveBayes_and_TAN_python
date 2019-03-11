#!/usr/bin/env python3.6
# chmod +x fileName.py

######################################################################################
# CS 760, Spring 2019
# HW 2
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# March of 2019
# Problem 2: Precision/recall Curve
######################################################################################

import json
import numpy as np 
import sys
import math
import matplotlib.pyplot as plt

from bayes_function import bayes


# Function to compute curve given resultes from predictions
def calculatePRCurve(resultsFromBayes):

	actual = resultsFromBayes[0]
	probabilities = resultsFromBayes[1]


	# thresholds = probabilities computed for each test instance
	thresholds = probabilities.copy()
	thresholds = list(set(thresholds)) # get ridded of repeated probabilities
	thresholds.sort() 	# sort probabilities

	# plot points
	precision = []
	recall = []

	for threshold in thresholds:
		TP = 0
		FP = 0
		FN = 0

		for i in range(len(probabilities)):
			if actual[i]:
				if probabilities[i] >= threshold:
					TP += 1
				else:
					FN += 1
			else:
				if probabilities[i] >= threshold:
					FP += 1
				# not necessary to compute TN

		precision.append(TP/(TP+FP))
		recall.append(TP/(TP+FN))

	return [precision, recall]




########################################################################
trainingSetPath = "../datasets/tic-tac-toe_train.json"
testSetPath = "../datasets/tic-tac-toe_test.json"

with open(trainingSetPath) as f:
    trainSet = json.load(f)

# Load test set
with open(testSetPath) as f:
    testSet = json.load(f)

trainingData = trainSet["data"]
testData = testSet["data"]

algorithm = "t"

[precision_t, recall_t] = calculatePRCurve(\
	bayes(trainingSetPath, testSetPath, algorithm, trainingData, testData))

[precision_n, recall_n] = calculatePRCurve(\
	bayes(trainingSetPath, testSetPath, "n", trainingData, testData))

fig = plt.figure()  # an empty figure with no axes
plt.plot(recall_n, precision_n, label = "Naive Bayes")
plt.plot(recall_t, precision_t, label = "TAN")
plt.xlabel('Recall (TPR)')
plt.ylabel('Precision')
plt.legend('Precision')
plt.title("Precision/Recall curve")
plt.grid(True)
plt.legend()
plt.show()




