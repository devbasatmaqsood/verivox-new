import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

# --- GLOBAL CACHE (Optimized) ---
# We store the exact folder where we found files to speed up subsequent lookups
FOUND_DIRS = set()

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

# --- TRAIN DATASET CLASS (2019 LA) ---
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
        # Training typically expects "flac" folder structure
        filepath = self.base_dir / "flac" / f"{key}.flac"
        try:
            X, _ = sf.read(str(filepath))
            X_pad = pad_random(X, self.cut)
            x_inp = Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y
        except Exception as e:
            return Tensor(np.zeros(self.cut)), self.labels[key]

# --- TURBO EVAL DATASET CLASS (Instant Lookup) ---
class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        # Root is typically /kaggle/input/avsspoof-2021/
        self.dataset_root = Path("/kaggle/input/avsspoof-2021/") 
        self.cut = 64600 
        self.missed_counts = 0 

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # 1. FAST CHECK: If we found files in a specific folder before, check there first!
        for d in FOUND_DIRS:
            p = d / f"{key}.flac"
            if p.exists():
                return self.load_audio(p, key)

        # 2. DEFINED PATHS: Check all known Kaggle Dataset Partitions
        # This prevents the slow "os.walk" search.
        paths_to_check = [
            self.base_dir / "flac" / f"{key}.flac",
            self.base_dir / f"{key}.flac",
        ]
        
        # Explicitly add parts 00 through 07 (covering the full dataset)
        parts = ["part00", "part01", "part02", "part03", "part04", "part05", "part06", "part07"]
        
        for part in parts:
            # Common Kaggle path structures
            p1 = self.dataset_root / f"ASVspoof2021_DF_eval_{part}" / "ASVspoof2021_DF_eval" / "flac" / f"{key}.flac"
            p2 = self.dataset_root / f"ASVspoof2021_DF_eval_{part}" / "flac" / f"{key}.flac"
            paths_to_check.append(p1)
            paths_to_check.append(p2)

        for p in paths_to_check:
            if p.exists():
                # Success! Save this directory to cache so next time it is instant
                FOUND_DIRS.add(p.parent)
                return self.load_audio(p, key)

        # 3. IF STILL MISSING
        if self.missed_counts < 10: 
            print(f"[ERROR] DF File missing: {key}.flac")
        self.missed_counts += 1
        return Tensor(np.zeros(self.cut)), key

    def load_audio(self, filepath, key):
        try:
            waveform, sample_rate = torchaudio.load(str(filepath))
            X = waveform.squeeze(0).numpy()
            X_pad = pad(X, self.cut)
            return Tensor(X_pad), key
        except:
            return Tensor(np.zeros(self.cut)), key