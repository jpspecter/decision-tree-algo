#!/usr/bin/python

import sys
import numpy as np
import pandas as pd

def sub_entropy(num):
  if(num == 0):
    return 0
  return num*np.log2(num)

def plurality_value(data):
  freq = data.iloc[:, -1].mode()

  
  class0_freq = freq.get(0, default = 0)
  class1_freq = freq.get(1, default = 0)
  class2_freq = freq.get(2, default = 0)

  if class0_freq >= class1_freq and class0_freq >= class2_freq:
    return 0, class0_freq
  elif class1_freq >= class0_freq and class1_freq >= class2_freq:
    return 1, class1_freq
  else:
    return 2, class2_freq

def calc_entropy(data):
  class_Values = data.iloc[:, -1]
  totalRows = (len(class_Values))
  if totalRows == 0:
    return 0
  entropy = 0
  for i in range(0, 3):
    p_i = ((class_Values==i).sum())/totalRows
    entropy += -sub_entropy(p_i)
  
  return entropy

def calc_conditional_entropy(data,attribute):
  conditional_entropy = 0;

  for i in range(0, 3):
    attr_subset = data[data[attribute] == i]
    conditional_entropy += (len(attr_subset)*calc_entropy(attr_subset))/len(data)
  
  return conditional_entropy

def best_attribute(data):
  
  sample_entropy = calc_entropy(data)
  max_info_gain = -1
  best_attr = ''

  for attr in data.columns[:-1]:
    info_gain_attr = sample_entropy - calc_conditional_entropy(data, attr)
    if info_gain_attr > max_info_gain:
      max_info_gain = info_gain_attr
      best_attr = attr

  return best_attr

def decision_tree_learning(data):
  mode_data = plurality_value(data)
  if len(data.index) == 0:
    return mode_whole_data[0]
  elif mode_data[1] == len(data.index):
    return mode_data[0]
  elif len(data.columns) == 1:
    return mode_data[0]
  else:
    best_attr = best_attribute(data)
    
    tree = {}
    tree[best_attr] = {}
    for i in range(0, 3):
      data_i = data[data[best_attr] == i]
      del data_i[best_attr]
      subtree = decision_tree_learning(data_i)
      tree[best_attr][i] = subtree
    
    return tree

def print_tree(tree, lvl = 0):
  if isinstance(tree, int):
    print(str(tree) + ' ')
    return
  print()
  for attr in tree.keys():
    for i in range(0, 3):
      print('| '*lvl + attr + ' = ' + str(i) + ' : ', end = '')
      print_tree(tree[attr][i], lvl+1)

  return 0

def tree_accuracy(tree, data):
  
  return 0

data_frame = pd.read_table('train4.dat')
class_counts = data_frame.iloc[:, -1].value_counts()
mode_whole_data = plurality_value(data_frame)

print(best_attribute(data_frame))
print(data_frame[data_frame[best_attribute(data_frame)] == 2])
#print(data_frame.iloc[:, -1].value_counts())
print_tree(decision_tree_learning(data_frame))
#def make_tree(dataset):