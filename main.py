#!/usr/bin/python

import sys
import numpy as np
import pandas as pd

#ID3 algorithm to generate the decision tree
def learning_algorithm(data):
 
  #simply returns the class instance with highest frequency  
  if len(data.index) == 0:
    return maxFrequency[0]

  #Stores the maximum occurence of class instance in the remaining dataset
  dataMax = maxFreqDataset(data)
  #if only one row instance is left, simply return the mode
  if len(data.iloc[:, -1].value_counts()) == 1:
    return dataMax[0]
  #if only one attribute is left
  elif len(data.columns) == 1:
    return dataMax[0]
  else:
    #Find the optimal attribute to split the tree
    attrBest = attrOptimal(data)
    
    #Placeholder for the final tree
    finalTree = {}
    finalTree[attrBest] = {}
    
    #Generating the tree by recursively finding the optimum subtree
    for i in range(0, 3):
      data_i = data[data[attrBest] == i]
      del data_i[attrBest]
      subtree = learning_algorithm(data_i)
      finalTree[attrBest][i] = subtree
    
    return finalTree

#This function satisfies the logarithmic condition mentioned in the prompt
def logCondition(num):
  if(num == 0):
    return 0
  return num*np.log2(num)

#This function finds the best attribute for splitting
def attrOptimal(data):
  
  #Find the entropy on splitting the data on each attribute
  entropySample = entropy(data)
  igMax = -1
  attrBest = ''

  #Finds the best attribute based on maximum information gain
  for attr in data.columns[:-1]:
    igAttr = entropySample - entropyCond(data, attr)
    if igAttr > igMax:
      igMax = igAttr
      attrBest = attr

  return attrBest

#Calculating the predictions based on the decision tree that I have created
def predictionCalc(finalTree, instance):
        
    if isinstance(finalTree, int):
      return finalTree

    attribute = list(finalTree.keys())[0]
    return predictionCalc(finalTree[attribute][instance[attribute]], instance)

#Returns the accuracy of the tree 
def accuracyCalc(finalTree, data):
  
  total = len(data.index)
  correct = 0
  for rIndex, row in data.iterrows():
     if(predictionCalc(finalTree, row.iloc[:-1]) == row.iloc[-1]):
        correct += 1
  return round((correct/total)*100, 2)

#Prints the decision tree to the console output
def treePrinter(finalTree, lvl = 0):
  if isinstance(finalTree, int):
    print(str(finalTree) + ' ')
    return
  
  print()
  
  for attr in finalTree.keys():
    for i in range(0, 3):
      print('| '*lvl + attr + '=' + str(i) + ':', end = '')
      treePrinter(finalTree[attr][i], lvl+1)

#Function to calculate the entropy of the dataset
def entropy(data):

  classes = data.iloc[:, -1]

  numRows = (len(classes))

  if numRows == 0:
    return 0
  entropy = 0

  for i in range(0, 3):
    probabilityI = ((classes==i).sum())/numRows
    entropy -= logCondition(probabilityI)
  
  return entropy

#Function to calculate the conditional entropy by splitting on each attribute
def entropyCond(data,attr):
  cEntropy = 0;

  for i in range(0, 3):
    subset = data[data[attr] == i]
    cEntropy += (len(subset)*entropy(subset))/len(data)
  
  return cEntropy

#Function to find the class instance with maximum frequency in the remaining dataset
#Used for tie-breaking as stated in the prompt rules
def maxFreqDataset(data):
    classes = data.iloc[:, -1]
    
    #array to hold the frequency of different class values
    classFreq = classes.value_counts()
    
    modes = data.iloc[:, -1].mode()
  
    classModes = modes.values
      
    tie_breaker = countClass.loc[classModes].sort_index() 
    classPlural = int(tie_breaker.idxmax())
  
    
    return classPlural, classFreq.get(classPlural)

'''    
#Throws an error if number of arguments is not equal to 3 (python file + training file + test file)
if(len(sys.argv) != 3):
    raise ValueError('Please enter exactly two arguments: ' + str(sys.argv))

#Getting the training and testing file from console and putting it into a dataframe
dfTraining = pd.read_table(str(sys.argv[1]))
dfTesting = pd.read_table(str(sys.argv[2]))
'''

dfTraining = pd.read_table('train4.dat')
dfTesting = pd.read_table('test4.dat')

#Stores the frequency of various class instances
countClass = dfTraining.iloc[:, -1].value_counts()

#Stores the class instance with the highest frequency
maxFrequency = maxFreqDataset(dfTraining)

#Applying the ID3 algorithm to generate a tree
resultTree = learning_algorithm(dfTraining)

#Printing the tree
treePrinter(resultTree)

#Printing the accuracy values
print('Accuracy on training set: ' + str(accuracyCalc(resultTree, dfTraining)))
print('Accuracy on testing set: ' + str(accuracyCalc(resultTree, dfTesting)))




