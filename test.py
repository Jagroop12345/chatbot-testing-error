import json
import pickle
import numpy as np
import random
import nltk

ignore_words = ['?', '!',',','.', "'s", "'m"]

import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model("./chatbot_model.h5")


intents = json.loads(open('./intents.json').read())
words = pickle.load(open("./words.pkl",'rb'))
classes = pickle.load(open('./classes.pkl',"rb"))

def preprocess_user_input(user_input):
    input_words_token_1 = nltk.word_tokenize(user_input)
    input_words_token_2 = get_stem_words(input_words_token_1,ignore_words)
    input_words_token_3 = sorted(list(set(input_words_token_2)))

    print(input_words_token_1)
    print(input_words_token_2)
    print(input_words_token_3)

    bag=[]
    bag_of_words=[]
    for word in words:
        #print(word)
        if word in input_words_token_3:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
        
    bag.append(bag_of_words)
    print(bag_of_words)

def bot_class_prediction(user_input):
    inp=preprocess_user_input(user_input)
    prediction = model.predict(inp)

    print(prediction)

    prediction_class_label = np.argmax(prediction[0])
    print(prediction_class_label)
    return prediction_class_label


def bot_response(user_input):
    pr_label = bot_class_prediction(user_input)
    pr_class = classes[pr_label]

    for i in intents["intents"]:
        if i['tag'] == pr_class:
            br = random.choice(i["responses"])
            return br
        
print("hi I am stella, How may I help you?")

while True:
    user_input = input("Type your message: ")
    print("user_input", user_input)

    response = bot_response(user_input)
    print('response', response)

# #preprocess_user_input("what help you provide?")
# bot_class_prediction("what help you provide?")