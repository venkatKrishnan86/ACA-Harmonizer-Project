from utils import Song
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json

# Build chord matrix
templates = json.load(open('chord_templates.json'))
N = len(templates.keys())
chord_matrix = np.zeros((N, 12))
index_to_chord = {}
for i, key in enumerate(templates.keys()):
    chord_matrix[i] = templates[key]
    index_to_chord[i] = key
def chord_detect_chromagram(song: Song, timestamps):
    S = np.abs(librosa.stft(song.x, n_fft=4096))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=song.sample_rate)

    n_blocks = S.shape[1]
    block_size = song.x.shape[0] / n_blocks

    chord_indexes = np.zeros(len(timestamps))
    chord_names = []

    for tidx, stamp in enumerate(timestamps):
        chroma_idx = int(stamp * song.sample_rate / block_size)
        chord_scores = similarity(chroma[:, chroma_idx], chord_matrix)
        chord_idx = np.argmax(chord_scores)
        chord_indexes[tidx] = chord_idx
        chord_names.append(index_to_chord[chord_idx])

    return chord_indexes, chord_names
def similarity(vector, chord_matrix):
    N = chord_matrix.shape[0]
    out = np.zeros(N)
    for i in range(N):
        chord = chord_matrix[i]
        out[i] = np.dot(vector, chord)/(np.linalg.norm(vector)*np.linalg.norm(chord))
        # out[i] = np.correlate(vector, chord)
    return out

