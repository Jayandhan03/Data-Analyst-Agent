import sounddevice as sd
import numpy as np
import whisper

model = whisper.load_model("small")

DEVICE_ID = 2  # your microphone

def record_audio(duration=5, fs=16000):
    print("Listening...")
    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='float32',
        device=DEVICE_ID
    )
    sd.wait()
    audio = np.squeeze(recording)
    print("Audio shape:", audio.shape)
    return audio * 3  # boost volume

while True:
    audio = record_audio()
    print("Transcribing...")
    result = model.transcribe(audio)
    print("You said:", result["text"])
