import pyaudio
import speech_recognition
import speech_recognition as sr


def audioToSpeech():
    r1 = sr.Recognizer()
    r2 = sr.Recognizer()
    r3 = sr.Recognizer()

    with sr.Microphone() as source:
        r3.adjust_for_ambient_noise(source, duration=0.2)
        audio = r3.listen(source)

    speech = r2.recognize_google(audio)

    return speech
