import pyttsx3

# init engine
engine = pyttsx3.init()

# (optional) tweak speaking rate and volume
engine.setProperty("rate", 180)    # words per minute
engine.setProperty("volume", 0.9)  # 0.0 to 1.0

def speak(text: str):
    print("Speaking:", text)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    speak("Hello, I am your voice data analyst assistant. Ask me anything about your data.")
