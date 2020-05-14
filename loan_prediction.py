    # -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:44:15 2020

@author: Haravindan
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Importing the dataset
dataset = pd.read_csv('loan_data.csv')

col_names=dataset.columns.values

for i in range(len(col_names)):
	if pd.isnull(dataset[col_names[i]]).sum()==0:
		print()
	else:
		if i < 5 or i > 7:
			mode=dataset[col_names[i]].mode()
			dataset[col_names[i]].fillna(mode.loc[0], inplace=True)
		else:
			dataset[col_names[i]].fillna(dataset[col_names[i]].mean(), inplace=True)
    			
category_col =['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area','Loan_Status'] 
labelEncoder = preprocessing.LabelEncoder() 

# mapping_dict ={} 
for col in category_col: 
	dataset[col] = labelEncoder.fit_transform(dataset[col]) 

# 	le_name_mapping = dict(zip(labelEncoder.classes_, 
# 	labelEncoder.transform(labelEncoder.classes_))) 

# 	mapping_dict[col]= le_name_mapping 
# print(mapping_dict) 

X = dataset.iloc[:,0:11].values
Y = dataset.iloc[:,11:].values


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0) 

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

#Prediction
y_pred = classifier.predict(X_test)


print("Random Forest Classifier Accuracy is ", 
			accuracy_score(y_test,y_pred)*100 ) 

with open('rf_classifier.pickle', 'wb') as file:
	pickle.dump(classifier, file, pickle.HIGHEST_PROTOCOL)
    
    
  

