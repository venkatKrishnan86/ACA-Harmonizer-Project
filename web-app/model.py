import torch
import torch.nn as nn
import librosa
import json
from tqdm import tqdm

SR = 44100
HOP = 256
FRAMES = 6
BATCH_SIZE = 2048

class Predictor(nn.Module):
    def __init__(self, device='mps'):
        super(Predictor, self).__init__()  # Transpose as well
        self.device = torch.device(device)
        self.conv1 = nn.Conv2d(1, 2, (1, 3), 1, (0, 1), device=self.device)
        self.norm1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, (1, 3), 1, (0, 1), device=self.device)
        self.norm2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 8, (1, 3), device=self.device)
        self.norm3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 12, (1, 3), device=self.device)
        self.norm4 = nn.BatchNorm2d(12)
        self.FC = nn.Linear(288, 12, device=self.device)

    def forward(self, x):
        x = torch.relu(self.norm1(self.conv1(x)))
        x = torch.relu(self.norm2(self.conv2(x)))
        x = torch.relu(self.norm3(self.conv3(x)))
        x = torch.relu(self.norm4(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = torch.softmax(self.FC(x), 1)
        return x

def predict(model, audio, chroma_req = True, chord_templates:dict = json.load(open('../chord_templates.json')), sr = SR, hop = HOP, device = 'cpu'):
    if chroma_req:
        chroma = torch.Tensor(librosa.feature.chroma_cens(y=audio, sr = sr, hop_length=hop)).T.unsqueeze(0)
    else:
        chroma = audio
    with torch.no_grad():
        outputs = nn.functional.softmax(model(chroma), 1)
    min_val = torch.Tensor([10000 for _ in range(outputs.shape[0])]).to(device)
    min_key = ["" for _ in range(outputs.shape[0])]
    for key, val in chord_templates.items():
        out = torch.Tensor([torch.norm(torch.Tensor(val).to(device) - i) for i in outputs]).to(device)
        min_val = torch.min(torch.stack([min_val, out], dim=1), dim=1)
        for i, truth in enumerate(min_val.indices==1):
            if truth:
                min_key[i] = key
        min_val = min_val.values
    return min_key

def prediction(model, chroma, frame = 6):
    stack = []
    time = []
    model.eval()
    pred = predict(model, chroma[:, :, :, :frame], False)[0]
    prev_pred = pred
    dur = 1
    main_sub = 0
    for i in tqdm(range(frame, chroma.shape[3]-frame+1, frame)):
        model.eval()
        pred = predict(model, chroma[:, :, :, i:i+frame], False)[0]
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
    if(len(stack)==0):
        stack.append(pred)
        time.append(0.0)
    return stack, time

def get_chroma(file):
    audio, _ = librosa.load(file, sr = SR)
    chroma = torch.Tensor(librosa.feature.chroma_cens(y=audio, sr = SR, hop_length=HOP)).unsqueeze(0)
    return torch.stack([chroma,chroma])

def run_model(path):
    model = Predictor(device='cpu').to(torch.device('cpu'))
    model.load_state_dict(torch.load('../models/chord_predictor1.pth'))
    chroma = get_chroma(path)
    stack = prediction(model, chroma)
    return stack

