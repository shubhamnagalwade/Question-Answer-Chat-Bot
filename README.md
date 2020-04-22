# Deep learning question-Answer Chatbot
This Chat bot is based on Sequential model and SGD optimizer. 
As well as this model deploy on GUI created by using python language. 
Pass the question from user get the output from bot it more accurate. As well as store the new question if its new for Bot.
Model train on 200 epochs.

# Install Required Packages(must latest veriosn)
import nltk \
nltk.download() \
import json \
import pickle \
import sys \
import tensorflow as tf \
import keras \
from tensorflow.keras.models import Sequential \
from tensorflow.keras.layers import Dense, Activation, Dropout \
from nltk.stem import WordNetLemmatizer \
lemmatizer = WordNetLemmatizer() \
from tensorflow.contrib.keras.python.keras.optimizers import SGD \

# Update the Tensorflow 2.0 
!pip upgrade tensorflow or !pip install tensorflow2.0
