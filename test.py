import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import Song

song_file = './beatles.wav'

song = Song(song_file, duration = 30, hop_size=512)
tempo, beats = song.getBeatsAndTempo()
print("Tempo:",tempo)
print("Beat timestamps:",beats)
f0_arr, time_in_sec = song.trackPitchACF()
print(f0_arr.shape)

X = librosa.stft(song.x, hop_length=512)
X_mag = np.abs(X)
librosa.display.specshow(X_mag, x_axis='time', y_axis='linear', hop_length=512, sr = 44100)
plt.plot(time_in_sec, f0_arr)
plt.ylim(0,1000)
plt.show()
