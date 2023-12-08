import numpy as np
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import json
import os
from sklearn.model_selection import train_test_split
from scipy.signal.windows import hann
import argparse

seed = 1
torch.manual_seed(seed)
chord_templates:dict = json.load(open('./chord_templates.json'))
HOP = 256
SR = 44100
WIN_LENGTH = 2048
WINDOW = hann(WIN_LENGTH)

class ChordDetector(Dataset):
    def __init__(self, train:bool, chord_template:dict = json.load(open('./chord_templates.json')), data_location:str = './data/', sr = SR, hop = HOP, frame = 6):
        super(ChordDetector, self).__init__()
        self.chord_template = chord_template
        self.data = []
        self.sr = sr
        self.hop = hop
        for file in os.listdir(data_location):
            chord_true = torch.Tensor(self.chord_template[self._extract_chord_name(file)])
            y, sr = librosa.load(data_location+file, sr = sr)
            y = y[:sr*2]
            # y += np.random.randn(len(y))*0.002
            y = y/np.max(np.abs(y))
            chroma = torch.Tensor(librosa.feature.chroma_cens(y=y, sr = sr, hop_length=hop)).T
            for i in range(0, chroma.shape[0]-frame+1,frame):
                self.data.append((chroma[i:i+frame,:], chord_true))
        X_train, X_test, _, _ = train_test_split(self.data, self.data, test_size=0.2, random_state=seed)
        if train:
            self.data = X_train
        else:
            self.data = X_test

    def _extract_chord_name(self, file):
        main = file[:file.index('-')]
        if file[file.index('-')+1]=='D':
            return main+'dim'
        elif file[file.index('-')+1]=='A':
            return main+'aug'
        elif file[file.index('-')+2]=='i':
            return main+'m'
        return main
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class GRU(nn.Module):
    def __init__(self, input_size = 12, hidden_size = 256, num_layers = 2, num_classes = 12, bidirectional = True) -> None:
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True, bidirectional=bidirectional)
        if(bidirectional):
            self.fc = nn.Linear(hidden_size*2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if(self.bidirectional):
            h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = out[:,-1,:] # Since we only want the output of the last cell
        out = self.fc(out)
        return(out)

def evaluate(model, chromas, chord_templates:dict = chord_templates):
    with torch.no_grad():
        outputs = nn.functional.softmax(model(chromas), 2)[0]
    min_val = 120
    min_key = ''
    for key, val in chord_templates.items():
        out = torch.norm(torch.Tensor(val) - outputs)
        if min_val >= out:
            min_val = out
            min_key = key
    return min_key

def predict(model, audio, chroma_req = True, chord_templates:dict = json.load(open('./chord_templates.json')), sr = SR, hop = HOP):
    if chroma_req:
        chroma = torch.Tensor(librosa.feature.chroma_cens(y=audio, sr = sr, hop_length=hop)).T.unsqueeze(0)
    else:
        chroma = audio
    with torch.no_grad():
        outputs = nn.functional.softmax(model(chroma), 1)[0]
    min_val = 120
    min_key = ''
    for key, val in chord_templates.items():
        out = torch.norm(torch.Tensor(val) - outputs)
        if min_val >= out:
            min_val = out
            min_key = key
    return min_key

def prediction(model, chroma, frame = 6):
    stack = []
    time = []
    model.eval()
    pred = predict(model, chroma[:frame, :].unsqueeze(0), False)
    prev_pred = pred
    dur = 1
    main_sub = 0
    for i in tqdm(range(frame, chroma.shape[0]-frame+1, frame)):
        model.eval()
        pred = predict(model, chroma[i:i+frame, :].unsqueeze(0), False)
        if(pred != prev_pred):
            if(dur>10):
                if(len(stack)==0):
                    stack.append(prev_pred)
                elif(stack[-1]==prev_pred):
                    dur = 0
                    prev_pred = pred
                    continue
                else:
                    stack.append(prev_pred)
                if len(time)!=0:
                    time.append((i)*HOP/SR - main_sub)
                else:
                    main_sub = (i)*HOP/SR
                    time.append(0.0)
            dur = 0
            prev_pred = pred
        dur+=1
    stack, time

if __name__=="__main__":
    device = torch.device('cpu')
    model = GRU().to(device)
    model.load_state_dict(torch.load('./models/chord_detector.pth'))
    model.eval()

    parser = argparse.ArgumentParser("Chord Detector")
    parser.add_argument("audio_file_path")
    arg = parser.parse_args()

    y = librosa.load(arg.audio_file_path, sr=SR)[0]
    y = y/np.max(np.abs(y))
    chroma = torch.Tensor(librosa.feature.chroma_cens(y=y, sr = SR, hop_length=HOP)).T
    stack, time = prediction(model, chroma)
    print(np.array([stack, time]).T[:30])


