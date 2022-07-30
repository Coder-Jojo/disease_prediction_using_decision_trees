1. To scrap the data from internet, run the preprocess.py file
    i.      This will get the disease name from nhp.gov.in 
    ii.     Find all the symptoms from mayoclinic
    iii.    Clean and preprocess the data
    iv.     Create the dataset
    v.      Save the data in .pkl format

2. Now, run the decision_tree.py file
    i.      This will build the decision tree 
    ii.     First decision tree will be built from kaggle dataset and it will stored as json format file 
    iii.    If the preprocess file was run and a dataset was created then the decision tree for the same will also be created and saved as json format file

3. Now, run web.py
    i.      It will ask for patient information
    ii.     Then this will ask whether to use kaggle dataset or scraped dataset 
    iii.    It will ask questions based on the decision tree nodes
    iv.     It will compile all the answers, show the result and find the prevention for the disease
