#!/usr/bin/env python3.6
# chmod +x fileName.py

######################################################################################
# CS 760, Spring 2019
# HW 2
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# March of 2019
# Problem 2: PR Curve --- Aux function, P1, but as a function
######################################################################################

printResults = 0
debug = 0

import json
import numpy as np 
import sys
import math


# function to compute P(xi|xj,y)
def PXgivenXY(trainingData, y, xj, idx_j, xi, idx_i, features):
	countXjY = 0
	countXiXjY = 0
	for instance in trainingData:
		if instance[-1] == y:
			if instance[idx_j] == xj:
				countXjY += 1
				if instance[idx_i] == xi:
					countXiXjY += 1

	P = (countXiXjY+1)/(countXjY+len(features[idx_i][1]))

	return P

# function to determine if adding newedge to current MST will create a cycle
def cicle(edgesMST, newedge):

	# separate edges into connected groups
	groups = [[] for edge in edgesMST]
	for edge in edgesMST:
		for group in groups:
			if group == []:
				group.append(edge)
				break
			else:
				inserted = 0
				for element in group:
					if edge[0]==element[0] or edge[0]==element[1] or\
					edge[1]==element[0] or edge[1]==element[1]:
						group.append(edge)
						inserted = 1
						break
				if inserted: break

	# test if edge is going to create cicle inside group
	for group in groups:
		visited = []
		for edge in group:
			visited.append(edge[0])
			visited.append(edge[1])
		if newedge[0] in visited and newedge[1] in visited:
			return 1 # cicle created

	return 0 # no cicle --->>> add egde



# Receive arguments using sys ########################################################
def bayes(trainingSetPath, testSetPath, algorithm, trainingData, testData):

	#algorithm = sys.argv[3] # n = Naive Bayes, t = TAN
	#trainingSetPath = sys.argv[1]
	#testSetPath = sys.argv[2]

	# Load dataset #######################################################################

	with open(trainingSetPath) as f:
	    trainSet = json.load(f)

	# Load test set
	with open(testSetPath) as f:
	    testSet = json.load(f)

	features = testSet["metadata"]["features"][:-1]
	classes = testSet["metadata"]["features"][-1][-1]
	numberFeatures = len(features)

	#trainingData = trainSet["data"]
	#testData = testSet["data"]

	# Output features for Naive Bayes ####################################################
	if printResults:
		if algorithm == "n":
			for feature in features:
				print(feature[0]+" class")
			print("")

	# Calculate edge weights if TAN ######################################################

	if algorithm == "t":
		if debug: print("calculating weights...")
		# Store results in matrix
		weights = np.zeros((len(features),len(features)))
		for i in range(len(features)):
			for j in range(i+1,len(features)):
				# for each feature combination:
				# matrix for counts:
				countsXiXjY1 = np.zeros((len(features[i][1]),len(features[j][1])))
				countsXiXjY2 = np.zeros((len(features[i][1]),len(features[j][1])))
				countsXjY1 = np.zeros(len(features[j][1]))
				countsXjY2 = np.zeros(len(features[j][1]))
				countsXiY1 = np.zeros(len(features[i][1]))
				countsXiY2 = np.zeros(len(features[i][1]))
				countsY1 = 0
				countsY2 = 0
				countsTotal = 0
				for instance in trainingData:
					# make counts
					countsTotal += 1
					if instance[-1] == classes[0]:
						countsY1 += 1
						countsXjY1[features[j][-1].index(instance[j])] += 1
						countsXiY1[features[i][-1].index(instance[i])] += 1
						countsXiXjY1[features[i][-1].index(instance[i])]\
						[features[j][-1].index(instance[j])] += 1
					else:
						countsY2 += 1
						countsXjY2[features[j][-1].index(instance[j])] += 1
						countsXiY2[features[i][-1].index(instance[i])] += 1
						countsXiXjY2[features[i][-1].index(instance[i])]\
						[features[j][-1].index(instance[j])] += 1

				# compute weights
				weight = 0
				for i2 in range(len(features[i][1])):
					for j2 in range(len(features[j][1])):

						probIntersecY1 = (countsXiXjY1[i2][j2]+1)\
						/(countsTotal+len(features[i][1])*len(features[j][1])*2)
						probXiXjgivenY1 = (1+countsXiXjY1[i2][j2])\
						/(countsY1+len(features[i][1])*len(features[j][1]))
						probXigivenY1 = (1+countsXiY1[i2])\
						/(countsY1+len(features[i][1]))
						probXjgivenY1 = (1+countsXjY1[j2])\
						/(countsY1+len(features[j][1]))

						weight += probIntersecY1*\
						math.log2(probXiXjgivenY1/(probXigivenY1*probXjgivenY1))

						probIntersecY2 = (countsXiXjY2[i2][j2]+1)\
						/(countsTotal+len(features[i][1])*len(features[j][1])*2)
						probXiXjgivenY2 = (1+countsXiXjY2[i2][j2])\
						/(countsY2+len(features[i][1])*len(features[j][1]))
						probXigivenY2 = (1+countsXiY2[i2])\
						/(countsY2+len(features[i][1]))
						probXjgivenY2 = (1+countsXjY2[j2])\
						/(countsY2+len(features[j][1]))

						weight += probIntersecY2*\
						math.log2(probXiXjgivenY2/(probXigivenY2*probXjgivenY2))
				weights[i][j] = weight
				weights[j][i] = weight

		if debug: print("Weights calculated")


	# Find MST ###########################################################################
	# Both Kruskal's and Prim's methods implemented
	if algorithm == "t":

		Kruskal = 0
		Prim = not Kruskal
		if debug: print("calculating MST...")

		if Kruskal:
			edgesMST = []

			edge = np.unravel_index(np.argmax(weights, axis=None), weights.shape)
			weights[edge[0]][edge[1]] = 0
			edgesMST.append(edge)
			while len(edgesMST) < len(features)-1:
				# take max weight edge
				edge = np.unravel_index(np.argmax(weights, axis=None), weights.shape)
				weights[edge[0]][edge[1]] = 0
				# does it close a loop?
				#print(edgesMST)
				#print(edge)
				#print(cicle(edgesMST, edge))
				if not cicle(edgesMST, edge):
					edgesMST.append(edge)

				#if debug: print(edgesMST)

		if Prim:
			edgesMST = []
			verticesMST = [0] # initialize with feature 0

			edge = (0, np.argmax(weights[0], axis=None))
			weights[:,0] = -1
			weights[:,edge[1]] = -1
			edgesMST.append(edge)
			verticesMST.append(edge[1])
			if debug: print(edgesMST)
			if debug: print(verticesMST)
			if debug: print(weights)
			if debug: print("")

			while len(verticesMST) < len(features):
				# take max weight edge inside possible rows
				possibleMatrix = np.zeros((len(verticesMST),len(features)))
				for i in range(len(verticesMST)):
					possibleMatrix[i] = weights[verticesMST[i]]
				edge = np.unravel_index(np.argmax(possibleMatrix, axis=None), \
				possibleMatrix.shape)
				edge = (verticesMST[edge[0]], edge[1])
				verticesMST.append(edge[1])
				edgesMST.append(edge)
				weights[:,edge[1]] = -1

				if debug: print(edgesMST)
				if debug: print(verticesMST)
				if debug: print(possibleMatrix)
				if debug: print("")

			if debug: print(edgesMST)




		if debug: print("MST calculated")

	# Assign edge directions in MST ######################################################

	if algorithm == "t":

		if debug: print("Directing MST...")
		visited = []
		for i in range(len(edgesMST)):
			edge = edgesMST[i]
			if edge[1] in visited: # flip it
				edgesMST[i] = (edge[1],edge[0])
			visited.append(edge[0])
			visited.append(edge[1])

		if debug: print("MST directed")
		if debug: print(edgesMST)

	# Output features for TAN ############################################################
	if printResults:
		if algorithm == "t":
			for feature in features:
				print(feature[0]+" ",end = "")
				# find parents
				parents = []
				for edge in edgesMST:
					if features[edge[1]][0] == feature[0]:
						parents.append(edge[0])
				parents.sort()
				for parent in parents:
					print(features[parent][0]+" ",end = "")
				print("class")
			print("")

	# Estimating classes probability = P(Y) ##############################################
	# Assuming binary classification

	classesCount = np.array([0, 0])
	for instance in trainingData:
		if instance[-1] == classes[0]:
			classesCount[0] += 1
		else:
			classesCount[1] += 1

	probClass = (classesCount+1)/(np.sum(classesCount+1))

	# Estimating P(Xi|Y) #################################################################
	# ---->>>> For Naive Bayes

	# Count features occurences for both classes 
	# List with matrices, 1 by feature:
	# Matrix: row0 = (count event1 class1) ... (count eventk class1) 
	#         row1 = (count event1 class2) ... (count eventk class2) 

	# initialize list of zeros
	counts = []
	probFeatures = []
	for feature in features:
		counts.append(np.vstack((np.zeros(len(feature[-1])),np.zeros(len(feature[-1])))))
		probFeatures.append(np.vstack((np.zeros(len(feature[-1])),np.zeros(len(feature[-1])))))

	# count through trainSet
	for instance in trainingData:
		if instance[-1] == classes[0]: 
			row = 0
		else:
			row = 1

		for i in range(len(features)):
			 counts[i][row][features[i][-1].index(instance[i])] += 1

	# compute probabilities based on counts
	for i in range(len(features)):
		probFeatures[i][0] = np.divide(counts[i][0]+1,np.sum(counts[i][0]+1))
		probFeatures[i][1] = np.divide(counts[i][1]+1,np.sum(counts[i][1]+1))

	# Estimating P(Y|testSet instance x) #################################################

	correctedPredicted = 0
	# count through testSet
	prediction_vector = []
	actual_vector = []
	probabilities_vector = []
	for instance in testData:
		# P(classx|instance) = P(classx)P(instance|classx)
		#                      ----------------------------------------------
		#                      Sum[all classes]{ P(class')P(instance|class')}
		# P(instance|classx) = Prod[all features]{P(xi|classx)} <<<--- Naive Bayes

		# P(instance|classx) for both classes:
		probTestInstGivenClass = np.array([1., 1.])
		if algorithm == "n":
			for i in range(len(features)):
				# First class
				probTestInstGivenClass[0] = probTestInstGivenClass[0]*\
				probFeatures[i][0][features[i][-1].index(instance[i])]
				# Second class
				probTestInstGivenClass[1] = probTestInstGivenClass[1]*\
				probFeatures[i][1][features[i][-1].index(instance[i])]
		if algorithm == "t":
			i = edgesMST[0][0]
			# First class
			probTestInstGivenClass[0] =\
			probFeatures[i][0][features[i][-1].index(instance[i])]
			# Second class
			probTestInstGivenClass[1] =\
			probFeatures[i][1][features[i][-1].index(instance[i])]
			for edge in edgesMST:
				# First class
				probTestInstGivenClass[0] = probTestInstGivenClass[0]*\
				PXgivenXY(trainingData, classes[0], instance[edge[0]],\
				edge[0], instance[edge[1]], edge[1], features)
				# PXgivenXY(trainingData, y, xj, idx_j, xi, idx_i, features)
				# Second class
				probTestInstGivenClass[1] = probTestInstGivenClass[1]*\
				PXgivenXY(trainingData, classes[1], instance[edge[0]],\
				edge[0], instance[edge[1]], edge[1], features)

		# P(classx|instance) for both classes:
		probClassGivenInst = np.array([1., 1.])
		# First class
		probClassGivenInst[0] = probClass[0]*probTestInstGivenClass[0]\
		/(probClass[0]*probTestInstGivenClass[0]+probClass[1]*probTestInstGivenClass[1])
		# Second class
		probClassGivenInst[1] = probClass[1]*probTestInstGivenClass[1]\
		/(probClass[0]*probTestInstGivenClass[0]+probClass[1]*probTestInstGivenClass[1])

		# Predict Y ######################################################################
		prediction = classes[np.argmax(probClassGivenInst)]
		actual = instance[-1]

		if prediction == actual: correctedPredicted += 1

		# Prediction output
		if printResults:
			print(prediction+" "+actual+" "+"%.12f"%np.max(probClassGivenInst))


		prediction_vector.append(prediction)
		actual_vector.append(actual == classes[0])
		probabilities_vector.append(probClassGivenInst[0])


	# Print number of correct predictions
	if printResults:
		print("")
		print(str(correctedPredicted))
		print("")


	return [actual_vector, probabilities_vector]




