import librosa
import numpy as np
import torch
from torch import nn 
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
import json

SR = 44100
HOP = 256
FRAMES = 6

class Song:
    def __init__(
            self, 
            song_file_path, 
            start = 0,
            end = None,
            block_size = 2048, 
            hop_size = 128, 
            sample_rate = 44100
        ):
        self.song_file = song_file_path
        self.block_size = block_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.x, _ = librosa.load(song_file_path, duration = end, sr=sample_rate)
        self.x = self.x[start*self.sample_rate:]

    def _block_audio(
            self
    ):
        """
            Returns:
            blocks_list: (numOfBlocks, blockSize)
            timeInSec: Start time index of each block; Shape - (numOfBlocks,)
        """
        if self.x.shape[0] % self.hop_size == 0:
            num_blocks = self.x.shape[0] // self.hop_size
        else:
            num_blocks = self.x.shape[0] // self.hop_size + 1
        ret = np.zeros((num_blocks, self.block_size))
        time_in_sec = np.zeros(num_blocks)
        for i in range(num_blocks):
            end_time = min(i * self.hop_size + self.block_size, self.x.shape[0])
            dur = end_time - i * self.hop_size
            ret[i][:dur] = self.x[i * self.hop_size : end_time]
            time_in_sec[i] = i * self.hop_size / self.sample_rate
        return ret, time_in_sec


    def _comp_acf(
            self,
            inputVector, 
            is_normalized = True
    ):
        """
            Returns:
            r: Autocorrelation result; Shape - (Length of inputVector,)
        """
        if is_normalized and inputVector.any():
            r = [np.correlate(inputVector[i:], inputVector[:-i], mode = 'valid')/np.linalg.norm(inputVector)**2 if i>0 else np.correlate(inputVector, inputVector, mode = 'valid')/np.linalg.norm(inputVector)**2 for i in range(len(inputVector))]
        else:
            r = [np.correlate(inputVector[i:], inputVector[:-i], mode = 'valid') if i>0 else np.correlate(inputVector, inputVector, mode = 'valid') for i in range(len(inputVector))]
        return np.array(r)


    def _get_f0_from_acf(
            self,
            r,
            amp_cutoff = -5,
            freq_max = 1000
    ):
        """
            Inputs -
            r: Autocorrelation result
            fs: Sample Rate

            Returns:
            freq: The frequency at which maximum autocorrelation value is achieved
        """
        mini_idx = np.argmin(r)
        amp = 20*np.log10(max(r[mini_idx + np.argmax(r[mini_idx:])], 0))
        freq = self.sample_rate / (mini_idx + np.argmax(r[mini_idx:]))
        if amp>amp_cutoff and freq < freq_max:
            return self.sample_rate / (mini_idx + np.argmax(r[mini_idx:]))
        else:
            return 0
    
    def _complete_parallel(self, mat, is_normalized = True):
        return self._get_f0_from_acf(self._comp_acf(mat, is_normalized))
    
    def _fillGaps(self, f0):
        count = 0
        time = 25
        value = 0
        f0_new = f0
        for i in range(len(f0)):
            if(f0[i]==0):
                count+=1
            else:
                if(count>=time):
                    count = 0
                else:
                    f0_new[i-count:i] = np.ones(count)*value
                    count = 0
                    value = 0
            if(i!=len(f0)-1 and f0[i+1]==0 and count==0):
                value = f0[i]
        return f0_new

    def trackPitchACF(
            self,
            gaussian_sigma = 1
    ):
        """
            Inputs -
            x: Input audio numpy array of shape - (n_samples,)
            block_size: Size of the block or number of samples per block (FFT size)
            hop_size: Number of samples to be hopped for each block
            fs: Sample Rate

            Returns:
            f0: Fundamental Frequency as a function of time; Shape - (numOfBlocks,)
            timeInSec: Start time index of each block; Shape - (numOfBlocks,)
        """
        is_normalized = True
        mat, time_in_sec = self._block_audio()

        f0_arr = np.array(Parallel(n_jobs=10)(delayed(self._complete_parallel)(mat[i], is_normalized) for i in tqdm(range(mat.shape[0]))))
        f0_arr = self._fillGaps(f0_arr) # fill gaps below 250 ms
        f0_arr = gaussian_filter1d(f0_arr, gaussian_sigma) # gaussian smooth


        return f0_arr, time_in_sec
    
    def getBeatsAndTempo(
            self
        ):
        tempo2, _ = librosa.beat.beat_track(y = self.x, sr= self.sample_rate, hop_length=self.hop_size, units="time")
        beats = np.arange(0,30,60/np.round(tempo2))
        return tempo2, beats
    
    @staticmethod
    def convertFreq2Midi(
            freq_in_hz
        ):
        fa = 440
        try:
            a = len(freq_in_hz)
            res = np.zeros_like(freq_in_hz)
            for i, freq in enumerate(freq_in_hz):
                if freq != 0:
                    res[i] = 12 * np.log2(freq / fa) + 69
                else:
                    res[i] = 0
            return res
        except:
            if freq_in_hz != 0:
                return 12 * np.log2(freq_in_hz / fa) + 69
            else:
                return 0

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
    
def get_chroma(file):
    audio, _ = librosa.load(file, sr = SR)
    chroma = torch.Tensor(librosa.feature.chroma_cens(y=audio, sr = SR, hop_length=HOP)).unsqueeze(0)
    return torch.stack([chroma,chroma])

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

def create_sine(freq, dur, sr=44100):
    return np.sin(np.linspace(0, 2*np.pi*freq*dur, int(sr*dur)))
