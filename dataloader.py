import os
import librosa
import numpy as np
from typing import Union, List
from torch.utils import data
import random
import soundfile as sf
from librosa.util import find_files


class LibriTTSDataset(data.Dataset):
    def __init__(
        self,
        root: Union[str, List],
        default_fs: int=16000,
        length_in_seconds=8,
        num_data_per_epoch=40000,
        random_start_point=False,
        train=True
    ):
        self.train = train
        self.default_fs = default_fs
        self.length_in_seconds = length_in_seconds
        self.seg_len = int(length_in_seconds * default_fs)
        self.random_start_point = random_start_point
        self.num_data_per_epoch = num_data_per_epoch
        
        if isinstance(root, str):
            self.meta = find_files(root, ext=['wav'])
        else:
            self.meta = []
            for r in root:
                self.meta += find_files(r, ext=['wav'])

        print("Dataset volume:", len(self.meta))
        self.sample_data_per_epoch()
    
    def sample_data_per_epoch(self):
        if self.train:
            self.meta_selected = random.sample(self.meta, self.num_data_per_epoch)
        else:  # select fixed data when in validation or test
            self.meta_selected = self.meta[:self.num_data_per_epoch]
    
    def read_audio(self, filename, force_1ch=False, fs=None, start=0, stop=None):
        audio, fs_ = sf.read(filename, dtype=np.float32, start=start, stop=stop, always_2d=True)
        audio = audio[:, :1].T if force_1ch else audio.T
        if fs is not None and fs != fs_:
            audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs)
            return audio, fs
        return audio, fs_

    def save_audio(self, audio, filename, fs):
        if audio.ndim != 1:
            audio = audio[0] if audio.shape[0] == 1 else audio.T
        sf.write(filename, audio, samplerate=fs)
    
    def __getitem__(self, idx):
        fs = self.default_fs
        rng = np.random.default_rng(idx)
        
        speech_path = self.meta_selected[idx]
        speech = self.read_audio(speech_path, force_1ch=True, fs=fs)[0]

        orig_len = speech.shape[1]
        
        if self.length_in_seconds != 0:  # length_in_seconds=0 means no cut or padding
            seg_len = self.seg_len
            if seg_len < orig_len:
                start_point = rng.integers(0, orig_len-seg_len) if self.random_start_point else 0
                speech = speech[:, start_point: start_point + seg_len]
            elif seg_len > orig_len:
                pad_points = seg_len - orig_len
                speech = np.pad(speech, ((0, 0), (0, pad_points)), constant_values=0)
 
        scale = 0.9 / (np.max(np.abs(speech)) + 1e-12)

        speech = speech.squeeze(0) * scale

        return speech
    
    def __len__(self):
        return self.num_data_per_epoch
    

if __name__ == "__main__":
    import soundfile as sf
    from tqdm import tqdm 
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train.yaml')
        
    train_dataset = LibriTTSDataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    
    validation_dataset = LibriTTSDataset(**config['validation_dataset'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])
    
    print(len(train_dataloader), len(validation_dataloader))
    
    # root = "/work/user_data/xiaobin/Datasets/dataloader_samples/train_samples"
    # os.makedirs(root, exist_ok=True)
    for i, speech in enumerate(tqdm(train_dataloader)):
        # speech = speech[0].numpy().T
        # sf.write(f"{root}/fileid_{i}_clean.wav", speech, 16000)
        
        # if i == 9:
        #     break
        pass
    
    
    # root = "/work/user_data/xiaobin/Datasets/dataloader_samples/valid_samples"
    # os.makedirs(root, exist_ok=True)
    for i, speech in enumerate(tqdm(validation_dataloader)):
        # speech = speech[0].numpy().T
        # sf.write(f"{root}/fileid_{i}_clean.wav", speech, 16000)
        
        # if i == 9:
        #     break
        pass

        
        