import numpy as np
import librosa
from utils import Song

song_file = './beatles.wav'

song = Song(song_file, duration = 30)
print(song.getBeatsAndTempo())
f0_arr, time_in_sec = song.trackPitchACF()
print(f0_arr.shape)
