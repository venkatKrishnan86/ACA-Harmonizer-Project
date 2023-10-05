import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import Song

song_file = './vocals.wav'
HOP_SIZE = 128

song = Song(song_file, start = 0, end = 60, hop_size=HOP_SIZE)
tempo, beats = song.getBeatsAndTempo()
print("Tempo:",tempo)
print("Beat timestamps:",beats)
f0_arr, time_in_sec = song.trackPitchACF()
print(f0_arr.shape)

X = librosa.stft(song.x, hop_length=HOP_SIZE)
X_mag = np.abs(X)

plt.figure(figsize = (18,10))
librosa.display.specshow(X_mag, x_axis='time', y_axis='linear', hop_length=HOP_SIZE, sr = 44100)
plt.plot(time_in_sec, f0_arr)
plt.ylim(0,1000)
plt.show()
