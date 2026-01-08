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

# --- GLOBAL CACHE TO SAVE DISCOVERED PATHS ---
# This prevents scanning the disk over and over again.
DISCOVERED_DIRS = set()

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
        filepath = self.base_dir / "flac" / f"{key}.flac"
        try:
            X, _ = sf.read(str(filepath))
            X_pad = pad_random(X, self.cut)
            x_inp = Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y
        except Exception as e:
            return Tensor(np.zeros(self.cut)), self.labels[key]

# --- SMART EVAL/DEV DATASET CLASS ---
class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = 64600 
        self.missed_counts = 0 

    def __len__(self):
        return len(self.list_IDs)

    def find_file_dynamic(self, key):
        """Searches for the file in discovered dirs first, then scans the whole input."""
        filename = f"{key}.flac"
        
        # 1. Check cached directories first (Fast)
        for d in DISCOVERED_DIRS:
            test_path = d / filename
            if test_path.exists():
                return test_path
        
        # 2. If not found, run a deep search (Slow, but only runs once per new folder)
        # print(f"Deep searching for {filename}...") # Uncomment to debug
        for root, dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                found_path = Path(root) / filename
                found_dir = Path(root)
                # Add to cache so next time it's fast
                DISCOVERED_DIRS.add(found_dir)
                return found_path
                
        return None

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # 1. Try standard paths
        paths_to_check = [
            self.base_dir / "flac" / f"{key}.flac",
            self.base_dir / f"{key}.flac",
        ]
        
        filepath = None
        for p in paths_to_check:
            if p.exists():
                filepath = p
                break
        
        # 2. If standard paths fail, use the Smart Search
        if filepath is None:
            filepath = self.find_file_dynamic(key)

        # 3. If STILL not found, error out
        if filepath is None:
            if self.missed_counts < 3: 
                print(f"[ERROR] File missing: {key}.flac. Checked all input folders.")
            self.missed_counts += 1
            return Tensor(np.zeros(self.cut)), key

        try:
            waveform, sample_rate = torchaudio.load(str(filepath))
            X = waveform.squeeze(0).numpy()
            X_pad = pad(X, self.cut)
            return Tensor(X_pad), key
        except:
            return Tensor(np.zeros(self.cut)), key