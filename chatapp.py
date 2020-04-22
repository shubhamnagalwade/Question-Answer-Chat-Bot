# -*- coding: utf-8 -*-
"""chatapp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vkf4IdOWw5MbKB0iT-U8bMvAc6J1fhSC
"""

import nltk
import pickle
import numpy as np
import json
import random
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tensorflow.contrib.keras.python.keras.models import load_model
#from keras.models import load_model

model = load_model('/home/shubham/Machine Learning/FireBlaze/ChatBot/2_templet/chatbot_model.h5')

intents = json.loads(open('/home/shubham/Machine Learning/FireBlaze/ChatBot/2_templet/intents.json').read())

words = pickle.load(open('/home/shubham/Machine Learning/FireBlaze/ChatBot/2_templet/words.pkl','rb'))

classes = pickle.load(open('/home/shubham/Machine Learning/FireBlaze/ChatBot/2_templet/classes.pkl','rb'))

def clean_up_sentence(sentence):
  #tokenize - split words into array
  sentence_words = nltk.word_tokenize(sentence)
  #stemming - create short form of word
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
  return sentence_words

def bow(sentence, words, show_details=True):
  #tokenize the pattern
  sentecne_words = clean_up_sentence(sentence)
  #bow
  bag = [0] * len(words)
  for s in sentecne_words:
    for i,w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print("found in bags: %s"% w)
    return(np.array(bag))

def predict_class(sentence, model):
  #filter out prediction below thresahold
  p = bow(sentence, words, show_details=False)
  res = model.predict(np.array([p]))[0]
  ERROR_THRESHOLD = 0.25
  results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

  #sort  by strength by probability 
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append({"intent":classes[r[0]], "probability":str(r[1])})
    return return_list

#random responce
def getResponse(ints, intents_json):
  tag = ints[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if(i['tag'] == tag):
      result = random.choice(i['responses'])
      break
  return result

def chatbot_responce(text):
  ints = predict_class(text, model)
  res = getResponse(ints, intents)
  return res
'''
"""***GUI***"""

import tkinter
from tkinter import *

def send():
  msg = EntryBox.get("1.0",'end-1c').strip()
  EntryBox.delete("0.0",END)

  if msg != '':
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "You:"+msg+'\n\n')
    ChatLog.config(foreground='#442265', font=("Verdana",12))

    res = chatbot_response(msg)
    ChatLog.insert(END,"Bot:"+res+'\n\n')

    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

import tkinter

base = Tk()
base.title("Hello")
base.geometry("50*150")
base.resizable(width=FALSE, height=FALSE)

#create chat window
ChatLog = Text(base, bd=0, bg="white", heigt='8', width="50", font = "Arial")

ChatLog.config(state=DISABLE)

#Bind scrollbar to chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursr = "heart")
ChatLog['yscrollcommand'] = scrollbar.set

#create button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12",
                      height=5, bd=0, bg="#32de97", activebackground='#3c9a9b',
                      fg='#ffffff', command=send)
  
#create the box  to enter message
EntryBox = Text(base, bd=0, bg="white", width='29', height="5", font="Arial")

#EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
'''