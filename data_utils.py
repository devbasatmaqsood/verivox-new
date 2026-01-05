import os
import numpy as np
import soundfile as sf
import torch
import torchaudio 
from torch import Tensor
from pathlib import Path
from torch.utils.data import Dataset

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
            label = parts[-1]  # Robust: grabs last column (bonafide/spoof)
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
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = path(base_dir)
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


# OVERRIDE Dataset Class to fix path issues
class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = 64600 

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # EXTENDED PATH SEARCH (Checks DF locations)
        paths_to_check = [
            self.base_dir / f"flac/{key}.flac",
            self.base_dir / f"{key}.flac",
            self.base_dir / "ASVspoof2021_DF_eval/flac" / f"{key}.flac",
            # Handle Kaggle double-nesting (common issue)
            self.base_dir / "ASVspoof2021_DF_eval/ASVspoof2021_DF_eval/flac" / f"{key}.flac" 
        ]
        
        filepath = None
        for p in paths_to_check:
            if os.path.exists(p): 
                filepath = p
                break
        
        if filepath is None:
            # Return silence if missing (prevents crash, but check your paths if this happens often!)
            return Tensor(np.zeros(self.cut)), key

        try:
            waveform, sample_rate = torchaudio.load(str(filepath))
            X = waveform.squeeze(0).numpy()
            # Basic padding logic
            if X.shape[0] < self.cut:
                num_repeats = int(self.cut / X.shape[0]) + 1
                X = np.tile(X, (1, num_repeats))[:, :self.cut][0]
            else:
                X = X[:self.cut]
            return Tensor(X), key
        except:
            return Tensor(np.zeros(self.cut)), key