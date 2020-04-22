import nltk
#nltk.download()
import json
import numpy as np
import pickle
import sys
import random
#import tensorflow.contrib.eager as tfe 
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras.python.keras.models import Sequential
#from keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, Dropout


#from tensorflow.keras.layers import Input, Dense
#from keras.layers import Dense, Activation, Dropout
#from tensorflow.python.keras.layers import Input, Dense
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tensorflow.contrib.keras.python.keras.optimizers import SGD


words = []
classes = []
ignore_words = ['?','!']
documents = []
data_file = open('/home/shubham/Machine Learning/FireBlaze/ChatBot/2_templet/python-project-chatbot-codes/intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
  for pattern in intent['patterns']:
    #tokenize each word
    w = nltk.word_tokenize(pattern)
    words.extend(w)
    #add documents in the corpus
    documents.append((w, intent['tag']))

    #add to our classes list
    if intent['tag'] not in classes:
      classes.append(intent['tag'])



words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#sort classes
classes = sorted(list(set(classes)))

#documents = combination between patterns ans intents
#print(len(documents), "documents")

#classes = intents
#print(len(classes),"classes",classes)

#words = all words, vocabulary
#print(len(words), "unique lemmatized words",words)


pickle.dump(words, open('/home/shubham/Machine Learning/FireBlaze/ChatBot/2_templet/python-project-chatbot-codes/words.pkl','wb'))
pickle.dump(classes, open('/home/shubham/Machine Learning/FireBlaze/ChatBot/2_templet/python-project-chatbot-codes/classes.pkl','wb'))


#create our training data
training = []
#create an empty array for our output
output_empty = [0] * len(classes)
#training set, bag of words for each sentence 
for doc in documents:
  #for initialize our bag of words
  bag = []
  #list of tokenize words for the pattern
  pattern_words = doc[0]
  #lemmatize each word - create base word, in attempt to represent related words
  pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
  #create our bag of words array with 1, if word match found in current pattern
  for w in words:
    bag.append(1) if w in pattern_words else bag.append(0)
  
  output_row = list(output_empty)
  output_row[classes.index(doc[1])] = 1

  training.append([bag, output_row])


#shuffle the features and turn into np.array
random.shuffle(training)
training = np.array(training)

#create train and test lists. X-patterns, Y-intents
train_x = list(training[:,0])
train_y = list(training[:,1])
#print("Training data created")

#create a model Tensorflow

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]), ), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


#compile model. Stochastic gradient descent with nesterov accelerated gradient gives good result

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd)

#fitting and saving model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.h5',hist)

print("model created")
















