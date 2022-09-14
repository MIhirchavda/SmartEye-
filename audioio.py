import speech_recognition as sr
from gtts import gTTS
import os
import playsound


"""
method for speek
"""
count = 0
def speek(text):
    global count
    tts = gTTS(text= text, lang = "en")
    filename = "voice"+str(count)+".mp3"
    print("file")
    tts.save(filename)
    print("gen")
    #os.system("play voice.mp3")
    playsound.playsound(filename)
    print("1")
    os.remove(filename)
    count+=1

    print("fin")

"""
get input from mic
"""
def getAudio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('0')
        audio = r.listen(source, timeout=2, phrase_time_limit=5)
        print('1')
        said = ""
        try:
            print('2')
            said = r.recognize_google(audio)
            print('3')
            return said
        except Exception as e:
            return str(e)
