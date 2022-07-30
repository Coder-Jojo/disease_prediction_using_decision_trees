#Importing required libraries
import numpy as np
from math import *
from pprint import pprint
import pickle
import json

# importing kaggle dataset
data_dir = 'dataset.csv'
def load_data(name):
  file_data = open(name)
  arr = []
  for row in file_data:
      arr.append([str(x) for x in row.split(',')])
  
  return np.array(arr)

# removing unnecessary spaces and underscores
def make_data_consistent(data):
  temp = data.copy()

  for i in range(temp.shape[0]):
    for j in range(temp.shape[1]):
      temp[i][j] = data[i][j].replace(' ', '').replace('_', ' ')  
  
  temp[:, 0] = data[:, 0]
  
  return temp

# remove duplications from symptoms
def get_unique_symptoms(data):
  temp =  np.unique(np.array(data).flatten())
  return temp[temp != '']

def sym_index(uniqueSymptoms):
  symptomsDict = {}

  for i in range(uniqueSymptoms.shape[0]):
    symptomsDict[uniqueSymptoms[i]] = i

  return symptomsDict

# break dataset into disease and symptoms
def get_diseases_and_symptoms(data, symptomsDict):
  diseases = data[1:, 0]

  symptoms = np.zeros((diseases.shape[0], uniqueSymptoms.shape[0]))

  for i in range(diseases.shape[0]):
    disease = diseases[i]
    for symptom in data[i+1, 1:]:
      if symptom != '':
        symptoms[i][symptomsDict[symptom]] = 1
  
  return diseases, symptoms

# create a combined data set with labels and column heading
def append_symptom_with_diseases(diseases, symptoms):
  temp = uniqueSymptoms.copy()
  temp = temp.flatten().tolist()
  temp.append('diseases')
  temp = np.array(temp)  
  symptoms = np.c_[symptoms, diseases]
  return np.r_[np.array([temp]), symptoms]

# check whether the given dataset contains same label
def has_common_label(data):
  data = data[1:]
  temp = np.unique(data[:, -1])
  return len(temp) == 1

# find all the labels in the given dataset and their count
def find_class(data):
  data = data[1:]
  labels = data[:, -1]

  labelName, labelCount = np.unique(labels, return_counts = True)

  return [labelName.tolist(), labelCount.tolist()]

# split dataset based on the infomation gain
def split_dataset(data, label):
  symptomNames = data[0]
  index = data[0].tolist().index(label)
  data = data[1:, :]

  yes = []
  no = []
  yes.append(symptomNames)
  no.append(symptomNames)

  for a in data:
    if int(float(a[index])):
      yes.append(a)
    else:
      no.append(a)
  
  return np.array(yes), np.array(no)

# calculate entropy
def entropy(data):
  data = data[1:]
  labels = data[:, -1]
  total = len(labels)

  labelName, labelCount = np.unique(labels, return_counts=True)
  entropy = 0

  for count in labelCount:
    entropy += count * log(count/total, 2) / total  
  
  return -entropy

# calculate information gain
def information_gain(data, symptom):
  
  dataEntropy = entropy(data)
  yes, no = split_dataset(data, symptom)

  totalSize = len(data) - 1

  yesEntropy = entropy(yes)
  noEntropy = entropy(no)

  totalYesSize = len(yes) - 1
  totalNoSize = totalSize - totalYesSize

  return dataEntropy - (totalYesSize / totalSize) * yesEntropy  - (totalNoSize / totalSize) * noEntropy

# remove the column with highest information gain
def split_column(data, index):
  a = data[:, :index]
  b = data[:, index+1:]
  temp = np.c_[a, b]
  return np.array(temp)

# implementing decision tree
def decision_tree(data, minSamples = 10):

  if has_common_label(data) or len(data) - 1 < minSamples:
    return find_class(data)
  
  informationGains = []

  for i in range(len(data[0]) - 1):
    informationGains.append(information_gain(data, data[0][i]))
  
  index = np.array(informationGains).argmax()

  yes, no = split_dataset(data, data[0][index])

  yes = split_column(yes, index)
  no = split_column(no, index)

  question = 'do you have {}?'.format('_'.join(data[0][index].split()))
  tree = {question: []}

  yesPart = decision_tree(yes)
  noPart = decision_tree(no)

  tree[question].append(yesPart)
  tree[question].append(noPart)

  return tree

# predicting using the decision tree
def predict_class(symptoms, tree, symptomName):
  question = list(tree.keys())[0]
  symptom = question.split()[-1].replace('_', ' ')[:-1]

  index = symptomName.tolist().index(symptom)
  answer = int(float(symptoms[index]))

  if answer == 1:
    next = tree[question][0]
  else:
    next = tree[question][1]

  if not isinstance(next, dict):
    return next

  return predict_class(symptoms, next, symptomName)

# function to calculate accuracy
def accuracy():
  c = 0
  total = len(symptoms)
  for i in range(total):
    temp = predict_class(symptoms[i], tree, processedData[0])
    labelName = temp[0]
    labelCount = temp[1]
    if labelName[np.array(labelCount).argmax()] == diseases[i]:
      c += 1
  
  print('Accuracy:', str(int(c/total * 100))+'%')

# processing the kaggle dataset
print('processing kaggle dataset')
rawData = load_data(data_dir)[:, :-1]
print('dataset loaded')

data = make_data_consistent(rawData)
print('data cleaned!')

uniqueSymptoms = get_unique_symptoms(data[1:, 1:])
symptomsDict = sym_index(uniqueSymptoms)
diseases, symptoms = get_diseases_and_symptoms(data, symptomsDict)
processedData = append_symptom_with_diseases(diseases, symptoms)
print('data processed')

print('creating decision tree')
tree = decision_tree(processedData, 30)
print('decision tree is created')
accuracy()

json_object = json.dumps(tree, indent = 2)
with open("tree.json", "w") as outfile:
    outfile.write(json_object)
json_object = json.dumps(processedData[0, :-1].tolist(), indent = 2)
with open("symptoms_kaggle.json", "w") as outfile:
    outfile.write(json_object)
print('files saved')

# processing scraped dataset
print('processing scraped dataset')
try:
  with open('scraped_and_preprocessed_dataset.pkl', 'rb') as handle:
      data = pickle.load(handle)
  print('processed data loaded')

  print('creating decision tree')
  scrap_tree = decision_tree(data, 30)
  print('decision tree created')

  json_object = json.dumps(scrap_tree, indent = 2)
  with open("scraped_data_tree.json", "w") as outfile:
      outfile.write(json_object)
  json_object = json.dumps(data[0,:-1].tolist(), indent = 2)
  with open("symptoms_kaggle.json", "w") as outfile:
      outfile.write(json_object)
  print('files saved')
except:
  print('such file do not exist')
