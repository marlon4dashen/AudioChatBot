import tkinter as tk
from ChatBot import chat
from AudioChat import audioToSpeech
import pyttsx3


def speak(text):

    e = pyttsx3.init()
    e.say(text)
    e.runAndWait()


window = tk.Tk()
window.title("AI Chat Bot")


def handle_click():
    speech = audioToSpeech()
    msg_box.insert(tk.END, "You: " + speech)
    window.update()
    str = chat(speech)
    msg_box.insert(tk.END, "Bot: " + str)
    window.update()
    speak(str)


messages = tk.Frame(window)
scrollBar = tk.Scrollbar(messages)
scrollBar.pack(side=tk.RIGHT, fill=tk.Y)
msg_box = tk.Listbox(messages, height=15, width=50, yscrollcommand=scrollBar)
msg_box.pack(side=tk.LEFT, fill=tk.BOTH)
messages.pack()

msg_box.insert(tk.END, "Please talk: ")


button = tk.Button(master=window, text="Speak", command=handle_click)
button.pack(side=tk.BOTTOM)

window.mainloop()
