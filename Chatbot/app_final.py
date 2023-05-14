# things we need for NLP
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask import Flask, jsonify, request
#from flask_cors import CORS, cross_origin

import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import random
import time
import datetime as dt
import logging
import os
import json

from pymongo import MongoClient
from bson import ObjectId

import datetime
#from spellcheck import spelling


today = datetime.datetime.today()

data = pickle.load( open( "chatbot-data1.pkl", "rb" ) )
words = data['words']
classes = data['classes']


data = pickle.load( open( "chatbot-data1.pkl", "rb" ) )
words = data['words']
classes = data['classes']



dirname = os.path.dirname(os.path.abspath("__file__"))
#print(dirname)
filename_logs = os.path.join(dirname, 'error/')


if __name__ == "__main__":
    print("Welcome to pyMongo")
    client = MongoClient("mongodb://172.29.0.186:27017/")
    print(client)
    db = client['db_genio']
    collection = db['genio_chatbot']

    # import json
    # with open('intents.json') as json_data:
    #   intents = json.load(json_data)

    find_result = collection.find_one(ObjectId('630c8bba613ec72c79fbb53e'))

    # find_one = collection.find_and_modify()
    # print(find_result)
    intents = pd.DataFrame(find_result)


import tensorflow as tf
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
tf.compat.v1.keras.backend.set_session(sess)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# p = bow("Load bood pessure for patient", words)
# print (p)
# print (classes)



from tensorflow.python.framework import ops
ops.reset_default_graph()

# # reset underlying graph data
ops.reset_default_graph()

# Use pickle to load in the pre-trained model
global graph
#graph = tf.get_default_graph()

with open(f'chatbot-model1.pkl', 'rb') as f:
    model = pickle.load(f)


def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    #sentence = spelling(sentence)
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    
    return return_list


def response(sentence):
    # logging.info(str(sentence))
    A = []
    C = []
    results = classify_local(sentence)
    A.append(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
        
            for i in intents['intents']:
                # logging.basicConfig(level=logging.DEBUG, filemode='a', filename='aaa.log')
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    #A = logging.debug(str(A[0]))
                    # A[0]+ ' --- ' +(
                    B = (random.choice(i['responses']))
                    C.append(str(B))
                    D = A,C
                    for i in range(len(A)):
                        # time.sleep(1)
                        # logging.debug(A[i])
                        x = (str(today) + '- CHATBOT -' + ' DEBUG - '  + A[i] + '\n')

                        if str(C[i])==("Please repeat what you said! \n\n For further assistance, post your issue with your employee code in your city's WhatsApp Helpdesk group."):
                            y = (str(today) + '- CHATBOT -' + ' ERROR - '  + C[i] + '\n')

                        elif str(C[i])==("Sorry! I didn't get your request. \n\n For further assistance, post your issue with your employee code in your city's WhatsApp Helpdesk group."):
                            y = (str(today) + '- CHATBOT -' + ' ERROR - '  + C[i] + '\n')
                            
                        elif str(C[i])==("Please be more specific, it was hard for me to understand. \n\n For further assistance, post your issue with your employee code in your city's WhatsApp Helpdesk group."):
                            y = (str(today) + '- CHATBOT -' + ' ERROR - '  + C[i] + '\n')

                        else:
                            # logging.info(C[i])
                            y = (str(today) + '- CHATBOT -' + ' INFO - '  + C[i] + '\n')
                    # log = str(x + '---' + y)
                    # T = log.split('---')
                    T = open(filename_logs + time.strftime("%d %B %Y") + "-user_logs.log",'a')   
                    n = T.writelines(x) 
                    Q = open(filename_logs + time.strftime("%d %B %Y") + "-chatbot_logs.log",'a')   
                    n = Q.writelines(y)          
                    
                    # logging.warning(str(T))
                    #print(type(log))
                    # print(type(T))
                    # text_file = open("sample.txt",'w')
                    # n = text_file.writelines(log)
                    # text_file.close()
                    # a random response from the intent
                    # return print(B)
                    break
            return B
    # results = classify_local(sentence)
    # # if we have a classification then find the matching intent tag
    # if results:
    #     # loop as long as there are matches to process
    #     while results:
    #         for i in intents['intents']:
    #             # find a tag matching the first result
    #             if i['tag'] == results[0][0]:
    #                 # a random response from the intent
    #                 return print(random.choice(i['responses']))

    #         results.pop(0)


classify_local('Bob is slow')


def chatbot_response(msg):
    ints = classify_local(msg)
    res = response(msg)
    #print(res)
    #print(type(res))
    res_dict={"Message":str(res)}
    # '{"Messege":'+'"'+str(res)+'"}'
    res_json=json.dumps(res_dict)
    #print(type(res_json))
    #print(res_json)
    return res_json
#json_data = json.loads(chatbot_response(msg))
    
from flask import Flask, render_template, request

app = Flask(__name__)

@app.errorhandler(500)
def internal_error(exception):
    app.logger.error(exception)
    return render_template('500.html'), 500

if app.debug is not True:   
    import logging
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler( filename_logs + time.strftime("%d %B %Y") + '-error_500.log', maxBytes=1024 * 1024 * 100, backupCount=20)
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
    

#if __name__ == "__main__":
#    app.run()
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port="5080")

    
