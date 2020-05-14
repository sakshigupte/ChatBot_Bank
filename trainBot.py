from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from numpy import array
from random import shuffle
from json import load
from pickle import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def parseDump():
    with open('Bank_ChatBot.json') as json_data:
        intents = load(json_data)
    vocabulary = []
    classes = []
    queries = []
    intentList = []
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = word_tokenize(pattern)
            # add to our words list
            vocabulary.extend(w)
            # add to documents in our corpus
            queries.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
        intentList.append(intent)
            
    # stem and lower each word and remove duplicates
    vocabulary = [stemmer.stem(w.lower()) for w in vocabulary if w not in ignored]
    vocabulary = sorted(list(set(vocabulary)))
    classes = sorted(list(set(classes)))
    # create our training data
    training = []   
    output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for query in queries:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = query[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in vocabulary:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(query[1])] = 1

        training.append([bag, output_row])
    # shuffle our features and turn into np.array
    shuffle(training)
    training = array(training)
    # create train and test lists
    x_train = array(list(training[:,0]))
    y_train = array(list(training[:,1]))
    return x_train, y_train, vocabulary, classes, intents

def trainModel():
    # Build model
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(len(vocabulary), )))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    # Train Model
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=8, epochs=500, verbose=1)
    return model

ignored = ['?']
x_train, y_train, vocabulary, classes, intents = parseDump()
model = trainModel()

# Save trained model
model.save('trainedModel/FAQbot_model.h5')
# Pickle the extra files
with open('trainedModel/vars.pkl', 'wb') as f:
    dump([vocabulary, classes, ignored, intents], f)
print('Training successful, model saved!')
