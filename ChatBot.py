import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import json
import random
from tensorflow import keras
import pickle
import tkinter as tk
from AudioChat import *

ERROR_THRESHHOLD = 0.25

#preprocessing data

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

model = keras.Sequential()

model.add(keras.layers.Dense(8, input_shape=(None, len(training[0]))))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(8))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(len(output[0]), activation="softmax"))

try:
    model = keras.models.load_model("models.h5")
except:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    training = np.expand_dims(training, axis=0)
    output = np.expand_dims(output, axis=0)
    fitModel = model.fit(training, output, epochs=100, batch_size=8, verbose=1)
    # model.save("model.h5")




def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)

    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat(str):

    tmp = np.expand_dims(bag_of_words(str, words), axis=(0, 1))
    results = model.predict(tmp)

    res = labels[np.argmax(results[0])]
    print(res)

    for tg in data["intents"]:
        if tg["tag"] == res:
            responses = tg["responses"]
    k = random.choice(responses)

    return k



window = tk.Tk()
window.title("AI Chat Bot")

def handle_click():
    speech = audioToSpeech()
    msg_box.insert(tk.END, "You: " + speech)
    window.update()
    msg_box.insert(tk.END, "Bot: " + chat(speech))
    window.update()

messages = tk.Frame(window)
scrollBar = tk.Scrollbar(messages)
scrollBar.pack(side=tk.RIGHT, fill=tk.Y)
msg_box = tk.Listbox(messages, height=15, width=50, yscrollcommand=scrollBar)
msg_box.pack(side=tk.LEFT, fill=tk.BOTH)
messages.pack()

msg_box.insert(tk.END, "Please talk: ")


button = tk.Button(master=messages, text="Speak", command=handle_click)
button.pack(side=tk.BOTTOM)

window.mainloop()








