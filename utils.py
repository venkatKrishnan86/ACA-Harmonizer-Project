import librosa
import numpy as np
import math
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

BLOCK_SIZE = 2048
HOP_SIZE = 128
SAMPLE_RATE = 44100

class Song:
    def __init__(
            self, 
            song_file_path, 
            block_size = BLOCK_SIZE, 
            hop_size = HOP_SIZE, 
            sample_rate = SAMPLE_RATE
        ):
        self.song_file = song_file_path
        self.block_size = block_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def _block_audio(
            self,
            x,
            block_size = BLOCK_SIZE, 
            hop_size = HOP_SIZE, 
            fs = SAMPLE_RATE
    ):
        """
            Inputs -
            x: Input audio numpy array of shape - (n_samples,)
            block_size: Size of the block or number of samples per block (FFT size)
            hop_size: Number of samples to be hopped for each block
            fs: Sample Rate

            Returns:
            blocks_list: (numOfBlocks, blockSize)
            timeInSec: Start time index of each block; Shape - (numOfBlocks,)
        """
        if x.shape[0] % hop_size == 0:
            num_blocks = x.shape[0] // hop_size
        else:
            num_blocks = x.shape[0] // hop_size + 1
        ret = np.zeros((num_blocks, block_size))
        time_in_sec = np.zeros(num_blocks)
        for i in range(num_blocks):
            end_time = min(i * hop_size + block_size, x.shape[0])
            dur = end_time - i * hop_size
            ret[i][:dur] = x[i * hop_size : end_time]
            time_in_sec[i] = i * hop_size / fs
        return ret, time_in_sec


    def _comp_acf(
            self,
            input_vector, 
            is_normalized = True
    ):
        """
            Returns:
            r: Autocorrelation result; Shape - (Length of inputVector,)
        """
        vec_len = input_vector.shape[0]
        padded = np.zeros(vec_len * 2)
        padded[:vec_len] = input_vector
        res = np.zeros(vec_len)
        # print(vec_len)
        for i in range(vec_len):
            res[i] = sum(input_vector * padded[i : i + vec_len])
        if is_normalized:
            res = res / sum(input_vector * input_vector)
        # print(res.shape)
        return res


    def _get_f0_from_acf(
            self,
            r, 
            fs = SAMPLE_RATE
    ):
        """
            Inputs -
            r: Autocorrelation result
            fs: Sample Rate

            Returns:
            freq: The frequency at which maximum autocorrelation value is achieved
        """
        mini_idx = -1
        for i in range(r.shape[0]):
            if i+1 == r.shape[0]:
                mini_idx = i
                break
            if r[i] < r[i+1]:
                mini_idx = i
                break
        maxx = -10
        maxx_idx = -1
        for i in range(mini_idx, r.shape[0]):
            if r[i] > maxx:
                maxx = r[i]
                maxx_idx = i
        freq = fs / maxx_idx
        return freq


    @staticmethod
    def track_pitch_acf(
            self,
            x, 
            block_size = BLOCK_SIZE, 
            hop_size = HOP_SIZE, 
            fs = SAMPLE_RATE
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
        mat, time_in_sec = self._block_audio(x, block_size, hop_size, fs)
        f0_arr = np.zeros(time_in_sec.shape[0])
        for i in range(mat.shape[0]):
            acf = self._comp_acf(mat[i], is_normalized)
            f0_arr[i] = self._get_f0_from_acf(acf, fs)
        return f0_arr, time_in_sec
    
    @staticmethod
    def getBeatsAndTempo(
            audio_file = 'beatles.wav', 
            duration = 30, 
            sample_rate = SAMPLE_RATE, 
            hop_size = HOP_SIZE//2
        ):
        y, sr = librosa.load(audio_file, duration = duration, sr = sample_rate) # 30s audio
        tempo2, _ = librosa.beat.beat_track(y = y, sr= sr, hop_length=hop_size, units="time")
        beats = np.arange(0,30,60/np.round(tempo2))
        return tempo2, beats
    
    @staticmethod
    def convert_freq2midi(
            self, 
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

