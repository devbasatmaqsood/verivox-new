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

# --- GLOBAL CACHE ---
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
            # Fallback for training safety
            return Tensor(np.zeros(self.cut)), self.labels[key]

# --- EVAL DATASET CLASS (Strictly 2021 DF) ---
class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        # Identify root to find other DF parts
        self.dataset_root = self.base_dir.parent.parent 
        self.cut = 64600 
        self.missed_counts = 0 

    def __len__(self):
        return len(self.list_IDs)

    def find_file_dynamic(self, key):
        """Searches for the file ONLY in DF folders."""
        filename = f"{key}.flac"
        
        # 1. Check cached DF directories
        for d in DISCOVERED_DIRS:
            test_path = d / filename
            if test_path.exists():
                return test_path
        
        # 2. Deep search ONLY for DF keys
        # We only search folders that look like they belong to DF to be strict
        for root, dirs, files in os.walk("/kaggle/input"):
            # STRICT CHECK: Ensure we are only looking in DF folders
            if "DF_eval" in root and filename in files:
                found_path = Path(root) / filename
                found_dir = Path(root)
                DISCOVERED_DIRS.add(found_dir)
                return found_path
                
        return None

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # 1. Try provided path (DF)
        paths_to_check = [
            self.base_dir / "flac" / f"{key}.flac",
            self.base_dir / f"{key}.flac",
        ]

        # 2. Try known DF parts (Explicitly DF only)
        parts = ["part00", "part01", "part02", "part03", "part04", "part05"] 
        for part in parts:
            p1 = self.dataset_root / f"ASVspoof2021_DF_eval_{part}" / "ASVspoof2021_DF_eval" / "flac" / f"{key}.flac"
            p2 = self.dataset_root / f"ASVspoof2021_DF_eval_{part}" / "flac" / f"{key}.flac"
            paths_to_check.append(p1)
            paths_to_check.append(p2)

        filepath = None
        for p in paths_to_check:
            if p.exists():
                filepath = p
                break
        
        # 3. Dynamic Search (DF Only)
        if filepath is None:
            filepath = self.find_file_dynamic(key)

        if filepath is None:
            if self.missed_counts < 3: 
                print(f"[ERROR] DF File missing: {key}.flac")
            self.missed_counts += 1
            return Tensor(np.zeros(self.cut)), key

        try:
            waveform, sample_rate = torchaudio.load(str(filepath))
            X = waveform.squeeze(0).numpy()
            X_pad = pad(X, self.cut)
            return Tensor(X_pad), key
        except:
            return Tensor(np.zeros(self.cut)), key