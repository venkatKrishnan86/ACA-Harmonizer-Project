import librosa
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

class Song:
    def __init__(
            self, 
            song_file_path, 
            duration = None,
            block_size = 2048, 
            hop_size = 128, 
            sample_rate = 44100
        ):
        self.song_file = song_file_path
        self.block_size = block_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.x, _ = librosa.load(song_file_path, duration = duration, sr=sample_rate)  

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
            r
    ):
        """
            Inputs -
            r: Autocorrelation result
            fs: Sample Rate

            Returns:
            freq: The frequency at which maximum autocorrelation value is achieved
        """
        mini_idx = np.argmin(r)
        freq = self.sample_rate / (mini_idx + np.argmax(r[mini_idx:]))
        return freq
    
    def _complete_parallel(self, mat, is_normalized = True):
        return self._get_f0_from_acf(self._comp_acf(mat, is_normalized))

    def trackPitchACF(
            self
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

        f0_arr = Parallel(n_jobs=10)(delayed(self._complete_parallel)(mat[i], is_normalized) for i in tqdm(range(mat.shape[0])))
        
        return np.array(f0_arr), time_in_sec
    
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

