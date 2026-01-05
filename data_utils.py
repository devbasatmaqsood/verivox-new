import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path  # <--- Fixes the 'NameError: name path is not defined'

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            parts = line.strip().split(" ")
            key = parts[1]
            label = parts[-1]
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            parts = line.strip().split(" ")
            key = parts[1]
            file_list.append(key)
        return file_list
    else:
        # Development / Validation set
        for line in l_meta:
            parts = line.strip().split(" ")
            key = parts[1]
            label = parts[-1]
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

# --- TRAIN DATASET CLASS ---
class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = Path(base_dir) 
        self.cut = 64600 

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # Robust path construction
        filepath = self.base_dir / "flac" / f"{key}.flac"
        
        try:
            X, _ = sf.read(str(filepath))
            X_pad = pad_random(X, self.cut)
            x_inp = Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y
        except Exception as e:
            # Fallback if file is missing/corrupt
            # print(f"Error loading train file {filepath}: {e}") 
            return Tensor(np.zeros(self.cut)), self.labels[key]

# --- EVAL/DEV DATASET CLASS ---
class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = 64600 
        self.missed_counts = 0 

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # SEARCH LOGIC: Checks both LA and DF locations
        paths_to_check = [
            self.base_dir / "flac" / f"{key}.flac",
            self.base_dir / f"{key}.flac",
            self.base_dir / "ASVspoof2021_DF_eval" / "flac" / f"{key}.flac",
            self.base_dir / "ASVspoof2021_DF_eval" / f"{key}.flac",
            self.base_dir / "ASVspoof2021_DF_eval" / "ASVspoof2021_DF_eval" / "flac" / f"{key}.flac"
        ]
        
        filepath = None
        for p in paths_to_check:
            if os.path.exists(p): 
                filepath = p
                break
        
        if filepath is None:
            if self.missed_counts < 3: # Limit error prints
                print(f"[ERROR] File missing: {key}.flac in {self.base_dir}")
            self.missed_counts += 1
            return Tensor(np.zeros(self.cut)), key

        try:
            waveform, sample_rate = torchaudio.load(str(filepath))
            X = waveform.squeeze(0).numpy()
            X_pad = pad(X, self.cut)
            return Tensor(X_pad), key
        except:
            return Tensor(np.zeros(self.cut)), key