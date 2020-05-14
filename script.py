# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:22:04 2020

@author: Haravindan
"""
from flask import Flask, render_template, request, redirect
import numpy as np
import pickle


from flask import Flask, render_template, request, redirect
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from numpy import array, argmax
from tensorflow.keras import models
from pickle import load
from random import choice


# prediction function 
def ValuePredictor(to_predict_list):        #loan prediction
    to_predict = np.array(to_predict_list).reshape(1,11) 
    with open('rf_classifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    result = loaded_model.predict(to_predict) 
    return result[0] 

def getQueryFeatures(query):        #FAQ ChatBot
    queryTokens = word_tokenize(query)
    queryStems = sorted(list(set([stemmer.stem(w.lower()) for w in queryTokens if w not in ignored])))
    queryBag = []
    for w in vocabulary:
        queryBag.append(1) if w in queryStems else queryBag.append(0)
    queryBag = array(queryBag)
    return queryBag.reshape(1, len(vocabulary))

with open('trainedModel/vars.pkl', 'rb') as f:
    vocabulary, classes, ignored, intents = load(f)
intentsDict = {i['tag']: i['responses'] for i in intents['intents']}
model = models.load_model('trainedModel/FAQbot_model.h5')
stemmer = LancasterStemmer()

def get_response(query):
    queryBag = getQueryFeatures(query)
    model.predict(queryBag)
    idx = argmax(model.predict(queryBag))
    print(idx)
    return choice(intentsDict[classes[idx]])

query = "question goes here"
reply = "bot reply goes here!"





app = Flask(__name__)


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/loan_predication')
def loan_predication():
    return render_template('index1.html')       #loan predictiom index1 file

@app.route('/result', methods = ['POST'])       #loan prediction index1 file
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(float, to_predict_list)) 
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Loan Can be granted'
        else: 
            prediction ='Loan Cannot be granted'            
        return render_template("index1.html", prediction = prediction)
        


@app.route('/bot')
def bot():
    return render_template('index.html', questionAsked=query, response=reply)   #FAQ Bot index file

@app.route('/signup', methods = ['POST'])
def signup():
    global query
    global reply 
    query = request.form['question']
    response = get_response(query)
    print(response)
    reply = response
    return redirect('/bot')        
        

    
if __name__ == "__main__":
    app.run()