from utils import get_chroma, prediction, create_sine, Predictor
import numpy as np
import librosa
import torch
import json
import argparse
from scipy.io.wavfile import write
import os

parser = argparse.ArgumentParser("chord_predictor")
parser.add_argument("file_path")
args = parser.parse_args()

device = torch.device('cpu')
model = Predictor(device=device).to(device)

model.load_state_dict(torch.load('./models/chord_predictor1.pth'))
model.eval()

file = args.file_path
chroma = get_chroma(file)
stack = prediction(model, chroma)
audio, _ = librosa.load(file, sr=44100)

base_C = 440*(2**(3/12))
chord_templates = json.load(open('./chord_templates.json'))
chords = [base_C*(2**(np.where(np.array(chord_templates[i])==1)[0]/12)) for i in stack[0]]
prev_time = 0.0
chord_sound = []
for chord_freq, time in zip(chords[:-1], stack[1][1:]):
    chord_sound.extend(create_sine(chord_freq[0], time-prev_time) + create_sine(chord_freq[1], time-prev_time) + create_sine(chord_freq[2], time-prev_time))
    prev_time = time

if not os.path.is_dir('./audios'):
    os.mkdir('./audios')
write('audios/sample.wav', 44100, np.array(chord_sound)*0.06+audio[:len(chord_sound)])