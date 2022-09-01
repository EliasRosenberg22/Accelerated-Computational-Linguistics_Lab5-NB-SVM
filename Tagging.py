#Author: Elias Rosenberg
#Date: May 3, 2021
#Purpose: change the given tagging model to get pos tags for a sentence of unspecified length.
#Input: Training texts
#ouptut: accuracy stats and words of the input sentence with their pos tags.



# ============================================================================================
# Dartmouth College, LING48, Spring 2021
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Exercise 5.2: Support Vector Machines
#
# Code sources:
# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
# ============================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Read the file with the data
cimData = pd.read_csv("/Users/eliasrosenberg/PycharmProjects/CS72 Lab 5/cim-4pos-punct.csv")

# The "X" matrix has the variables that we'll use as predictors
# The "y" array has the labels that we want to predict
X = cimData.drop('tokenPOS', axis=1)
y = cimData['tokenPOS']

# Display shape of the matrix.
# print(X)


# Display examples of the feature vectors and the labels
print("\nfeature vector of the element zero of the training set:")
print(X.head())
print("\nlabel of the element zero of the training set:")
for i in range(5):
    print(str(i) + ": " + y[i])

# The OrdinalEncoder function converts categorical variables to
# numerical variables, so that we can use them in the regression.
# The LabelEncoder function does the same wih the labels.
# This will transform the y vector so that:
# TAM = 0   Verb = 1
encoderX = preprocessing.OrdinalEncoder()
tx = encoderX.fit_transform(X)
encoderY = preprocessing.LabelEncoder()
ty = encoderY.fit_transform(y)

# Display examples of the encoded feature vectors and the labels
print("\nordinal and label encoding for the training set:")
for i in range(5):
    print(str(i) + ": " + str(tx[i]) + "\t" + str(ty[i]))

print("\nConvert a label encoding back to its encoding in the CSV:")
print("0: " + str(encoderY.inverse_transform([0])))
print("1: " + str(encoderY.inverse_transform([1])))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tx, ty, test_size=0.10)

# Train the classifier with the training data
# Use the default RBF Kernel
# Use the default gamma= 1 / (n_features * X.var())
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

# Make predictions about the test data
y_pred = svclassifier.predict(X_test)

# Report the results of the test data
# y_test contains the actual labels for each of the test items
# y_pred contains the predicted labels for each of the test items
print("\n=== CONFUSION MATRIX ===")
print("0:TAM  1:V")
print(confusion_matrix(y_test, y_pred))
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))


#  prevToken token  postToken
# [ - ,      kia,   orana ]
def getPredictionFromList(listWord):
    listOfLists = [listWord]

    returnElement = ""

    try:
        listTransformed = encoderX.transform(listOfLists)
        y_pred = svclassifier.predict(listTransformed)
        y_predLabel = encoderY.inverse_transform(y_pred)
        returnElement = y_predLabel[0]
    except:
        returnElement = "<UNK>"

    return returnElement

#===========================================================================================================
print("\n=== PREDICTION ===")
print("Enter a sentence and press ENTER: ")
userInput = input()
print()

predPhrase = userInput
tokens = predPhrase.split(" ")

#print("here is the list of tokens")
#print(tokens)

if len(tokens) == 1:

    tempFeats = ["-", tokens[0], "-"]
    print(tokens[0] + " - " + getPredictionFromList(tempFeats))

elif len(tokens) == 2:

    tempFeats = ["-", tokens[0], tokens[1]]
    #print(tempFeats)
    print(tokens[0] + " - " + getPredictionFromList(tempFeats))

    tempFeats = [tokens[0], tokens[1], "-"]
    #print(tempFeats)
    print(tokens[1] + " - " + getPredictionFromList(tempFeats))


elif len(tokens) > 2: #code for when the number of tokens is more than a unigram or bigram
    for i in range(len(tokens)):
        if i == 0:
            tempFeats = ["-", tokens[i], tokens[i + 1]] #list for the first word in tokens
            #print(tempFeats)
            print(tokens[i] + " - " + getPredictionFromList(tempFeats))
        elif i == len(tokens)-1:
            tempFeats = [tokens[i - 1], tokens[i], "-"] #list for the last word in tokens
            #print(tempFeats)
            print(tokens[i] + " - " + getPredictionFromList(tempFeats))
        else:
            tempFeats = [tokens[i - 1], tokens[i], tokens[i + 1]] #list for all other words in the middle of the tokens list
            #print(tempFeats)
            print(tokens[i] + " - " + getPredictionFromList(tempFeats))



