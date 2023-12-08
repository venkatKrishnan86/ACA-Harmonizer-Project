import numpy as np
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchsummary import summary
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import json
import os
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import soundfile as sf
from utils import GRU

SR = 44100
HOP = 256
FRAMES = 6
BATCH_SIZE = 2048

chord_detector = GRU()
chord_detector.load_state_dict(torch.load('./models/chord_detector.pth'))
chord_detector.eval()

class MelChordDataset(Dataset):
    def __init__(
            self, 
            data_location = "../../../../Music Technology/Datasets/musdb18hq/",
            out_location = "../../../../Music Technology/Datasets/musdb18hq/",
            frames_per_chord = 6,
            train = True,
            write_data = False
        ):
        super(MelChordDataset).__init__()
        if write_data:
            self._write_chords_and_audio(data_location, out_location, train)
        self.frames_per_chord = frames_per_chord
        vocals_y = []
        vocals_chroma = []
        chord_templates:dict = json.load(open('./chord_templates.json'))
        act_chord_data = []
        self.data_location= data_location+"chunks_vocal/"
        self.out_location= out_location+"chunks_chord/"
        if not train:
            self.data_location[:-1]+="_test/"
            self.out_location[:-1]+="_test/"

        for i in range(len(os.listdir(self.data_location))): # 100
            with open(self.out_location+"chord_"+str(i), "rb") as fp:
                chord_data = pickle.load(fp)
            act_chord_data.append(torch.Tensor(np.array([np.array(chord_templates[i]) for i in chord_data])))
            vocals_y.append(librosa.load(self.data_location + 'vocal_'+str(i)+'.wav', sr=SR)[0])
            vocals_chroma.append(torch.Tensor(librosa.feature.chroma_cens(y=vocals_y[-1], sr = SR, hop_length=HOP)).T)
        
        # act_chord_data[i]: Shape: (num_chords[i], 12)
        # vocals_chroma[i]: Shape: (num_frames[i], 12)
        # num_chords[i] = (num_frames[i] // frames_per_chord)

        self.data = []
        self._create_data(act_chord_data, vocals_chroma)
    
    def _create_data(self, chord_data, chroma_data):
        for (chroma, chords) in zip(chroma_data, chord_data):
            for i in range(0, chroma.shape[0]-self.frames_per_chord, self.frames_per_chord):
                block_chroma = chroma[i:i+self.frames_per_chord,:]
                block_chord = chords[i//self.frames_per_chord]
                if(block_chroma.any()):
                    self.data.append((block_chroma, block_chord))

    def _write_chords_and_audio(
            self, 
            data_location, 
            out_location, 
            train = True
        ):
        if train:
            data_location = data_location+"train/"
        else:
            data_location = data_location+"test/"
        folders = os.listdir(data_location)
        count = 0

        for folder in folders:
            if not os.path.isdir(data_location+folder):
                continue
            mixture_y, _ = librosa.load(data_location + '/' + folder + '/mixture.wav', sr=SR)
            vocals_y, _ = librosa.load(data_location + '/' + folder + '/vocals.wav', sr=SR)
            mixture_y = mixture_y/np.max(np.abs(mixture_y))
            vocals_y = vocals_y/np.max(np.abs(vocals_y))

            mixture_chroma = torch.Tensor(librosa.feature.chroma_cens(y=mixture_y, sr = SR, hop_length=HOP)).T
            chunk_length = FRAMES
            nchunks = mixture_chroma.shape[0] // chunk_length # no padding

            if train:
                if not os.path.isdir(out_location+'chunks_chord'):
                    os.mkdir(out_location+'chunks_chord')
                if not os.path.isdir(out_location+'chunks_vocal'):
                    os.mkdir(out_location+'chunks_vocal')
            else:
                if not os.path.isdir(out_location+'chunks_chord_test/'):
                    os.mkdir(out_location+'chunks_chord_test')
                if not os.path.isdir(out_location+'chunks_vocal_test/'):
                    os.mkdir(out_location+'chunks_vocal_test')

            # Get chords from mixture chroma
            chord_stack, time = MelChordDataset.prediction(chord_detector, mixture_chroma)
            frame_num = np.array([int(i/((HOP/SR)*6)) for i in time])
            chord_stack = np.array([frame_num, chord_stack]).T
            chords = []
            for prev, curr in zip(chord_stack[:-1], chord_stack[1:]):
                frame_diff = int(curr[0]) - int(prev[0])
                chords.extend([prev[1] for _ in range(frame_diff)])
            chords.extend([chord_stack[-1][1] for _ in range(nchunks - len(chords))])

            if train:
                with open(out_location+"chunks_chord/chord_"+str(count), "wb") as fp:
                    pickle.dump(chords, fp)
                sf.write(out_location + 'chunks_vocal/vocal_' + str(count)+'.wav', vocals_y, SR)
            else:
                with open(out_location+"chunks_chord_test/chord_"+str(count), "wb") as fp:
                    pickle.dump(chords, fp)
                sf.write(out_location + 'chunks_vocal_test/vocal_' + str(count)+'.wav', vocals_y, SR)
            count+=1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    @staticmethod
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
    
    @staticmethod
    def prediction(model, chroma, frame = 6):
        stack = []
        time = []
        model.eval()
        pred = MelChordDataset.predict(model, chroma[:frame, :].unsqueeze(0), False)
        prev_pred = pred
        dur = 1
        main_sub = 0
        for i in tqdm(range(frame, chroma.shape[0]-frame+1, frame)):
            model.eval()
            pred = MelChordDataset.predict(model, chroma[i:i+frame, :].unsqueeze(0), False)
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
        return stack, time
    
class Predictor(nn.Module):
    def __init__(self, device = 'mps'):
        super(Predictor, self).__init__() # Transpose as well
        self.device = torch.device(device)
        self.conv1 = nn.Conv2d(1, 2, (1,3), 1, (0, 1), device=self.device)
        self.norm1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, (1,3), 1, (0, 1), device=self.device)
        self.norm2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 8, (1,3), device=self.device)
        self.norm3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 12, (1,3), device=self.device)
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
    
def predict(model, audio, chroma_req = True, chord_templates:dict = json.load(open('./chord_templates.json')), sr = SR, hop = HOP, device = 'cpu'):
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

if __name__ == "__main__":
    # train_data = MelChordDataset(train = True, write_data = False)
    # test_data = MelChordDataset(train = False, write_data = False)
    # torch.save(train_data, './data/final/train_data.pt')
    # torch.save(test_data, './data/final/test_data.pt')
    train_data = torch.load('./data/final/train_data.pt')
    test_data = torch.load('./data/final/test_data.pt')

    train_loader = DataLoader(
        train_data,
        BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        test_data,
        BATCH_SIZE,
        shuffle=False
    )

    device = torch.device('mps')
    model = Predictor(device=device).to(device)
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 5e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)

    best_weights = copy.deepcopy(model.state_dict())
    max = 0
    val_acc = 0
    train_acc = 0
    chord_templates:dict = json.load(open('./chord_templates.json'))
    for epoch in range(num_epochs):
        device = 'mps'
        model = model.to(device)
        model.train()
        for i, (chromas,chords) in tqdm(enumerate(train_loader)):
            chromas = torch.transpose(chromas,1,2).unsqueeze(1).to(device)
            chords = chords.to(device)

            preds = model(chromas)
            loss = criterion(preds, chords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step() # Decaying learning rate per 25 epochs by 0.2 times
        print(f'Epoch {epoch+1}/{num_epochs}; Loss = {loss.item():.6f}; LR = {scheduler.get_last_lr()}')
        if (epoch+1)%5==0:
            with torch.no_grad():
                n_samples = 0
                n_correct = 0
                model.eval()
                device = 'cpu'
                model = model.to(device)
                for chromas, chords in tqdm(test_loader):
                    chromas = torch.transpose(chromas,1,2).unsqueeze(1).to(device)
                    chords = chords.to(device)
                    pred_outputs1 = model(chromas)
                    prediction = predict(model, chromas, chroma_req=False)
                    predictions = torch.tensor(np.array([chord_templates[i] for i in prediction])).to(device)
                    n_samples += chords.shape[0]*chords.shape[1]
                    n_correct += (predictions == chords).sum().item()
                val_acc = n_correct/n_samples * 100

                if (max < (n_correct/n_samples * 100)):
                    print('SAVED MODEL WEIGHTS')
                    max = val_acc
                    best_weights = copy.deepcopy(model.state_dict())

                if (epoch+1)%100==0:
                    n_samples = 0
                    n_correct = 0
                    
                    for chromas, chords in tqdm(train_loader):
                        chromas = torch.transpose(chromas,1,2).unsqueeze(1).to(device)
                        chords = chords.to(device)
                        pred_outputs1 = model(chromas)
                        prediction = predict(model, chromas, chroma_req=False)
                        predictions = torch.tensor(np.array([chord_templates[i] for i in prediction])).to(device)
                        n_samples += chords.shape[0]*chords.shape[1]
                        n_correct += (predictions == chords).sum().item()
                    train_acc = n_correct/n_samples * 100
                    print(f'Train Accuracy: {train_acc:.2f}%')
            print(f'Dev Accuracy: {val_acc:.2f}%')
            print("-"*20)
    
    torch.save(model.to('cpu').state_dict(), './models/chord_predictor1.pth')
