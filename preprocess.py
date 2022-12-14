# -*- coding: utf-8 -*-
"""preprocess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZVOPIid1FZ3qHuyQnAPXa16aOEU7OZQU
"""
# importing necessary libraries
import requests
import re
import time
import numpy as np
import pickle
import nltk
import json
import random
from bs4 import BeautifulSoup
nltk.download("popular")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet 
from itertools import combinations
from googlesearch import search

# function to save data in pickle file
def save_data(name, target):
  with open(name + '.pkl', 'wb') as handle:
   pickle.dump(target, handle, protocol=pickle.HIGHEST_PROTOCOL)

# function to extract data from pickle file
def load_data(name):
  with open(name + '.pkl', 'rb') as handle:
    return pickle.load(handle)

# scrap disease names form nhp.gov.in
def get_disease_names():
  diseaseNames = []
  for i in range(26):
      url = 'https://www.nhp.gov.in/disease-a-z/'+ chr(ord('a') + i)
      time.sleep(1)
      
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')

      allDiseases = soup.find('div', class_='all-disease')

      for disease in allDiseases.find_all('li'):
          diseaseNames.append(disease.get_text().strip())
  
  return diseaseNames

# find the symptom node in beautiful soup
def find_symptom_node(query):
  try:
    response = requests.get(query, timeout=3)
  except Exception: 
    print('some problem occurred in response from server')
    time.sleep(5)
    pass
  
  time.sleep(1)
  soup = BeautifulSoup(response.content, 'html.parser')

  heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
  heading = None
  for tags in soup.find_all(heading_tags):
    if(tags.text.lower().strip() == 'symptoms'):
      heading = tags
      break
  
  return heading

# scrap symptoms for all the diseases
def find_symptoms(diseases):

  disease_symptom = {}

  for disease in diseases:

    q = None

    query = disease + ' mayoclinic'
    for link in search(query, stop=10,pause=0.5): 
      if re.search(r'https://www.mayoclinic.org/diseases-conditions', link):
        q = link
        break
    
    print('link:', q)
    if q == None:
      time.sleep(1)
      continue
    
    heading = find_symptom_node(q)
      
    if heading == None:
      for _ in range(2):
        heading = find_symptom_node(q)
        if heading != None:
          break

      if heading == None:
        print('Could not find data for', disease, '\n')
      continue

    symptoms = []
    heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

    heading = heading.next_sibling
    while heading.name not in heading_tags:
      if heading.name == 'ul':
        symptoms += heading.find_all('li')
      heading = heading.next_sibling

    temp = symptoms.copy()
    symptoms = []

    for symptom in temp:
      sentence = symptom.text.split('.')[0]
      if len(sentence.split(' ')) <= 6:
        sentence = sentence.replace('.', '').replace(';', ',')
        sentence = re.sub(r'<[^<]+?>',', ', sentence) # All the tags
        sentence = ' '.join([x for x in sentence.split() if x != ','])
        symptoms.append(sentence)
    
    print('disease:', disease, '\nSymptoms:', symptoms)
    if len(symptoms) > 0:
      disease_symptom[disease] = symptoms
    else:
      print('Could not find understandable symptoms for', disease, 'because sentences were complex')

    print()

  return disease_symptom

# clean the scraped data
def clean_data(disease_symptom):
  
  stop_words = stopwords.words('english')
  lemmatizer = WordNetLemmatizer()
  splitter = RegexpTokenizer(r'\w+')

  cleaned_disease_sym = {}

  for key, value in disease_symptom.items():
    value = ','.join(value)
    symptoms = []

    for sym in re.sub(r"\[\S+\]", "", value).lower().split(','):
      sym = sym.strip()
      if len(sym) > 0 and sym != 'none':
        symptoms.append(sym)

    if len(symptoms) == 0:
      continue
    
    temp = []
    for sym in symptoms:
      sym = sym.replace('-',' ').replace("'",'').replace('(','').replace(')','')
      sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym) if word not in stop_words and not word[0].isdigit()])
      temp.append(sym)

    
    cleaned_disease_sym[key] = temp
  
  return cleaned_disease_sym

# find synonyms for the given term
def find_synonyms(term):
    synonyms = []
  
    response = requests.get('https://www.thesaurus.com/browse/' + term)
    soup = BeautifulSoup(response.content,  "html.parser")

    try:
        container=soup.find('section', {'class': 'MainContentContainer'}) 
        row=container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        for sym in row.find_all('li'):
          synonyms.append(sym.get_text())
    except:
        None
    for synonym in wordnet.synsets(term):
        synonyms += synonym.lemma_names()
        
    return set(synonyms)

def get_all_symptoms(dis_sym):
  symp = []
  for key, val in dis_sym.items():
    symp += val

  return np.unique(symp).tolist()

# find synonym for given symptom
def find_simptom_synonym(disSym):

  symp = get_all_symptoms(disSym)
  
  symptom_syn_pair = dict()
  
  for sym in symp:
      temp = sym.split()

      synonyms = set()

      for lenght in range(1, len(temp) + 1):
          for combination in combinations(temp, lenght):
              synonyms.update(find_synonyms(' '.join(combination)))
              
      synonyms.add(sym)
      synonyms = set(' '.join(synonyms).replace('_', ' ').lower().split())
      symptom_syn_pair[sym] = synonyms

  return symptom_syn_pair

# merge all the symptoms which are similar
def merge_similar_symptoms(dis_syn, sym_syn):
  symptoms = get_all_symptoms(dis_syn)
  n = len(symptoms)
  final_symptom = dict()

  for i in range(n):
    for j in range(i+1, n):
      s1 = symptoms[i]
      s2 = symptoms[j]
      a = sym_syn[s1]
      b = sym_syn[s2]

      if len(a.intersection(b)) / len(a.union(b)) > 0.5:

        if s1 in final_symptom.keys():
          final_symptom[s2] = final_symptom[s1]
        else:
          final_symptom[s2] = s1

  return final_symptom

# process the scrap data
def preprocessed_disease_symptoms_pair(clean_data, final_symptom):
  
  preprocessed = dict()
  for disease, symptoms in clean_data.items():
    preprocessed[disease] = []
    for symptom in symptoms:
      if symptom in final_symptom.keys():
        preprocessed[disease].append(final_symptom[symptom])
      else:
        preprocessed[disease].append(symptom)
  
  return preprocessed

# create dataset required by the decision tree algorithm
def creating_dataset(data):
  all_symptoms = get_all_symptoms(data)

  top_row = all_symptoms.copy()
  top_row.append('diseases')
  top_row = np.array(top_row)

  cnt = 0

  for disease, symptoms in data.items():
    for lenght in range(int(len(symptoms) * .7), len(symptoms) + 1):
      for combination in combinations(symptoms, lenght):
        cnt += 1

  processed = np.zeros((cnt, len(all_symptoms)))
  labels = []
  i = 0
  for disease, symptoms in data.items():
    for lenght in range(int(len(symptoms) * .7), len(symptoms) + 1):
      for combination in combinations(symptoms, lenght):
        for symptom in combination:
          processed[i][all_symptoms.index(symptom)] = 1
        
        i += 1
        labels.append(disease)

  processed = np.c_[processed, np.array(labels)]
  processed = np.r_[[top_row], processed]
  return processed

# executions of the functions
diseases = np.unique(get_disease_names())
print('a total of', diseases.shape[0], 'diseases found\n\n\n')
save_data('diseases', diseases)

print('randomly selecting 50 diseases from the given dataset\n\n\n')
n = len(diseases)
subset_diseases = []
for i in random.sample(range(0, n-1), 50):
  subset_diseases.append(diseases[i])

disease_symptom = find_symptoms(subset_diseases)
print('disease symptom pairs created\n\n\n')
save_data('disease_symptom', disease_symptom)

cleaned_disease_sym = clean_data(disease_symptom)
print('disease symptom pairs cleand\n\n\n')

sim_syn = find_simptom_synonym(cleaned_disease_sym)
print('symptoms synonyms created\n\n\n')

similar_symptoms = merge_similar_symptoms(cleaned_disease_sym, sim_syn)
print('similar symptoms dictionary created!\n\n\n')

data = preprocessed_disease_symptoms_pair(cleaned_disease_sym, similar_symptoms)
print('Data scraped and preprocessed')

scraped_and_preprocessed_dataset = creating_dataset(data)
print('Dataset for decision tree created')

save_data('scraped_and_preprocessed_dataset', scraped_and_preprocessed_dataset)
json_object = json.dumps(get_all_symptoms(data), indent = 2)
with open("scraped_data_symptom.json", "w") as outfile:
    outfile.write(json_object)