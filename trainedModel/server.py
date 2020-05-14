from flask import Flask, render_template, request, redirect
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from numpy import array, argmax
from tensorflow.keras import models
from pickle import load
from random import choice

def getQueryFeatures(query):
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
    return choice(intentsDict[classes[idx]])

query = "question goes here"
reply = "bot reply goes here!"



app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html', questionAsked=query, response=reply)

@app.route('/signup', methods = ['POST'])
def signup():
	global query
	global reply
	query = request.form['question']
	response = get_response(query)
	print(response)
	reply = response
	return redirect('/')

if __name__ == "__main__":
    app.run()
