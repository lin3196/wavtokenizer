import os
import librosa
import pandas as pd
import numpy as np
from scipy.io import loadmat
from typing import List
from torch.utils import data
from pymatreader import read_mat
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pymatreader')


class BBLDataset(data.Dataset):
    def __init__(
        self,
        speech_root: str,
        speech_datasets: List[str],
        default_fs: int=16000,
        length_in_seconds=8,
        num_data_tot=720000,  # no used at the moment
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
        
        # For speech meta info
        self.speech_meta = []
        for name in speech_datasets:
            csv_dir = os.path.join(speech_root, f"{name}_mat")
            csv_path = os.path.join(csv_dir, f"{name}.csv")
            assert os.path.exists(csv_path), csv_dir
            
            csv_info = pd.read_csv(csv_path)
            
            meta_info = pd.DataFrame()
            path_depth = len(csv_info.dstFilename.iloc[0].split('/'))
            # print(f"Speech dataset {name}: {path_depth}")
            
            if path_depth == 3:
                meta_info['filePath'] = [os.path.join(csv_dir, mat_name) for mat_name in csv_info.dstFilename]
            elif path_depth == 4:
                meta_info['filePath'] = [os.path.join(os.path.dirname(csv_dir), mat_name) for mat_name in csv_info.dstFilename]
            else:
                raise ValueError(f"Mera file {csv_path} has unmatched dstFilenames")
            meta_info['fileFs'] = csv_info.fileFs
            
            self.speech_meta.append(meta_info)
        self.speech_meta = pd.concat(self.speech_meta, axis=0).sample(frac=1, random_state=0)
        
        assert len(self.speech_meta) >= num_data_per_epoch, len(self.speech_meta)
        print("Dataset volume:", len(self.speech_meta))
        self.sample_data_per_epoch()
    
    def sample_data_per_epoch(self, epoch=0):
        if self.train:
            self.speech_meta_selected = self.speech_meta.sample(self.num_data_per_epoch, ignore_index=True, random_state=epoch)
        else:  # select fixed data when in validation or test
            self.speech_meta_selected = self.speech_meta.iloc[:self.num_data_per_epoch]
    
    def read_audio_from_mat(self, mat_file, fs):
        # with h5py.File(mat_file) as f:
        #     audio = f['audio'][:]
        # audio = loadmat(mat_file)
        # print(mat_file)
        
        try:
            data = read_mat(mat_file)
        except:
            print("Failed to read mat file:", mat_file)
            
        audio = list(data.values())[-1]  # ambiguous key, e.g. audio_YFUFHQQGY, audio_BYDT08U4Z
        
        if audio.ndim > 1:
            audio = audio[:, 0]

        if fs < self.default_fs:
            print("[Warning] Valid frequency band smaller than default sampling rate!")
        elif fs > self.default_fs:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.default_fs)
        
        return audio[None, :]  # (1, L)
    
    def __getitem__(self, idx):
        s_meta = self.speech_meta_selected.iloc[idx]
        
        speech = self.read_audio_from_mat(s_meta.filePath, s_meta.fileFs)
        
        orig_len = speech.shape[-1]
        
        rng = np.random.default_rng(idx)
        
        if self.length_in_seconds != 0:  # length_in_seconds=0 means no cut or padding, use in test
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
    
    config = OmegaConf.load('configs/cfg_train_16k.yaml')
        
    train_dataset = BBLDataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    
    validation_dataset = BBLDataset(**config['validation_dataset'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])
    
    print(len(train_dataloader), len(validation_dataloader))
    
    root = "/work/user_data/xiaobin/Datasets/dataloader_samples/train_samples"
    os.makedirs(root, exist_ok=True)
    for i, speech in enumerate(tqdm(train_dataloader)):
        speech = speech[0].numpy().T
        sf.write(f"{root}/fileid_{i}_clean.wav", speech, 16000)
        
        if i == 9:
            break
        # pass
    
    
    root = "/work/user_data/xiaobin/Datasets/dataloader_samples/valid_samples"
    os.makedirs(root, exist_ok=True)
    for i, speech in enumerate(tqdm(validation_dataloader)):
        speech = speech[0].numpy().T
        sf.write(f"{root}/fileid_{i}_clean.wav", speech, 16000)
        
        if i == 9:
            break
        # pass

        
        