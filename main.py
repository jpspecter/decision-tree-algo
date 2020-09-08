#!/usr/bin/python

import sys
import numpy as np
import pandas as pd

def learning_algorithm(data):
  
  if len(data.index) == 0:
    return maxFrequency[0]

  dataMax = plurality(data)
  if dataMax[1] == len(data.index):
    return dataMax[0]
  elif len(data.columns) == 1:
    return dataMax[0]
  else:
    attrBest = attrOptimal(data)
    
    finalTree = {}
    finalTree[attrBest] = {}
    for i in range(0, 3):
      data_i = data[data[attrBest] == i]
      del data_i[attrBest]
      subtree = learning_algorithm(data_i)
      finalTree[attrBest][i] = subtree
    
    return finalTree

def logCondition(num):
  if(num == 0):
    return 0
  return num*np.log2(num)

def attrOptimal(data):
  
  entropySample = entropy(data)
  igMax = -1
  attrBest = ''

  for attr in data.columns[:-1]:
    igAttr = entropySample - entropyCond(data, attr)
    if igAttr > igMax:
      igMax = igAttr
      attrBest = attr

  return attrBest

def predictionCalc(finalTree, instance):
    pointerTree = finalTree
        
    while not isinstance(pointerTree, int):
        test_attr = list(pointerTree.keys())[0]
        subtree = pointerTree[test_attr]
        pointerTree = subtree[instance[test_attr]]
    
    return pointerTree

def accuracyCalc(finalTree, data):
  
  total = len(data.index)
  correct = 0
  for row_num, row in data.iterrows():
     if(predictionCalc(finalTree, row.iloc[:-1]) == row.iloc[-1]):
        correct += 1
  return round((correct/total)*100, 2)

def treePrinter(finalTree, lvl = 0):
  if isinstance(finalTree, int):
    print(str(finalTree) + ' ')
    return
  
  print()
  
  for attr in finalTree.keys():
    for i in range(0, 3):
      print('| '*lvl + attr + '=' + str(i) + ':', end = '')
      treePrinter(finalTree[attr][i], lvl+1)

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

def entropyCond(data,attr):
  cEntropy = 0;

  for i in range(0, 3):
    subset = data[data[attr] == i]
    cEntropy += (len(subset)*entropy(subset))/len(data)
  
  return cEntropy

def plurality(data):

  classes = data.iloc[:, -1]
  
  #array to hold the frequency of different class values
  classFreq = classes.value_counts()

  modes = data.iloc[:, -1].mode()
  
  classModes = modes.values
      
  tie_breaker = countClass.loc[classModes].sort_index() 
  classPlural = int(tie_breaker.idxmax())
  
    
  return classPlural, classFreq.get(classPlural)

#if(len(sys.argv) != 2):
#    raise ValueError('Please enter exactly two arguments: ' + str(sys.argv))

#dfTraining = pd.read_table(str(sys.argv[0]))
#dfTesting = pd.read_table(str(sys.argv[1]))

dfTraining = pd.read_table('train.dat').sample(n=800)
dfTesting = pd.read_table('test.dat')

countClass = dfTraining.iloc[:, -1].value_counts()

maxFrequency = plurality(dfTraining)

resultTree = learning_algorithm(dfTraining)

treePrinter(resultTree)
print('Accuracy on training set: ' + str(accuracyCalc(resultTree, dfTraining)))

print('Accuracy on testing set: ' + str(accuracyCalc(resultTree, dfTesting)))




