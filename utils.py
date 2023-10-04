import librosa
import numpy as np

def getBeatsAndTempo(
        audio_file = 'beatles.wav', 
        duration = 30, 
        sample_rate = 44100, 
        hop_size = 64
    ):
    y, sr = librosa.load(audio_file, duration = duration, sr = sample_rate) # 30s audio
    tempo2, _ = librosa.beat.beat_track(y = y, sr= sr, hop_length=hop_size, units="time")
    beats = np.arange(0,30,60/np.round(tempo2))
    return tempo2, beats
