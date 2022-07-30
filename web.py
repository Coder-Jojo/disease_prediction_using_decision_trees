# importing required libraries
from pywebio.input import *
from pywebio.output import *
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import re
import time 
from googlesearch import search

# function to load data
def load_data(name):
  file_data = open(name)
  a = json.load(file_data)
  return a

# function to choose between dataset
def choose_tree():
    answer = radio('Please select the decision tree', options=['Tree made from kaggle dataset', 'Tree made from scraped dataset'])
    
    if answer == 'Tree made from kaggle dataset':
        return load_data('tree.json'), load_data('symptoms_kaggle.json')
    else:
        return load_data('scraped_data_tree.json'), load_data('scraped_data_symptom.json')

# function to predict the disease
def predict_class(tree, symptomName):
    question = list(tree.keys())[0]
    symptom = question.split()[-1].replace('_', ' ')[:-1]

    answer = radio(question.replace('_', ' '), options=['Yes', 'No'])

    if answer == 'Yes':
        answer = 1
    else:
        answer = 0

    if answer == 1:
        next = tree[question][0]
    else:
        next = tree[question][1]

    if not isinstance(next, dict):
        return next

    return predict_class(next, symptomName)

# function to find the location of prevention node
def find_prevention_node(query):
  try:
    response = requests.get(query, timeout=3)
  except Exception: 
    time.sleep(5)
    pass
  
  time.sleep(1)
  soup = BeautifulSoup(response.content, 'html.parser')

  heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
  heading = None
  for tags in soup.find_all(heading_tags):
    if(tags.text.lower().strip() == 'prevention'):
      heading = tags
      break
  
  return heading

# function to find the preventions for the disease
def find_prevention(disease):
    try:
        q = None

        query = disease + ' mayoclinic'
        for link in search(query, stop=10,pause=0.5): 
            if re.search(r'https://www.mayoclinic.org/diseases-conditions', link):
                q = link
                break

        if q == None:
            return -1

        heading = find_prevention_node(q)
            
        if heading == None:
            for _ in range(2):
                heading = find_prevention_node(q)
                if heading != None:
                    break

                if heading == None:
                    return -1

        put_html('<h3>Prevention</h3>', sanitize=False, scope=None, position=- 1)
        heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        heading = heading.next_sibling
        while heading and heading.name not in heading_tags:
            put_html(str(heading), sanitize=False, scope=None, position=- 1)
            heading = heading.next_sibling
    
    except:
        pass

# function to display output
def output(result, patient):
    disease_label = result[0]
    disease_count = result[1]


    table = [['Title', 'Details']]
    for key, value in patient.items():
        table.append([key, value])

    put_html('<h3>Patient Details</h3>', sanitize=False, scope=None, position=- 1)
    put_table(table)

    ans = [['You might be suffering from']]
    for i in range(len(disease_count)):
        ans.append([disease_label[i]])
    put_table(ans)

    find_prevention(disease_label[0])

# function for taking patients details
def take_user_details():
    data = input_group("Patient Details", [
    input('Name', name='name'),
    input('Age', name='age', type=NUMBER),
    radio('Gender', name='gender', options=['Male', 'Female']),
    input('Height (in cm)', name='height', type=FLOAT),
    input('Weight (in kg)', name='weight', type=NUMBER),
    textarea('Allergies', rows=3, name='allergies', placeholder='Peanuts, rose smell, etc...'),
    checkbox("User Term", name='agree', options=['I give permission to store my details in the database'])
    ])

    if data['agree'] == ['I give permission to store my details in the database']:
        popup('Message', 'Your details have been saved')
    return data

# driving code
put_html('<h1>Medical App</h1>', sanitize=False, scope=None, position=- 1)
user_details = take_user_details()

tree, symptom = choose_tree()
result = predict_class(tree, np.array(symptom))

output(result, user_details)