import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    for line in l_meta:
        try:
            parts = line.strip().split()
            # We check if the line has at least 2 parts (Speaker ID and Utterance ID)
            if len(parts) >= 2:
                # In both 2019 and 2021 datasets, the Key (Filename) is at index 1
                key = parts[1]
                
                # Smart Label Search: Look for the label anywhere in the line
                if "bonafide" in parts:
                    label = "bonafide"
                elif "spoof" in parts:
                    label = "spoof"
                else:
                    # If no label is found:
                    # If it's the blind evaluation set (is_eval), we might accept it without a label.
                    # But since you are using metadata now, we expect labels.
                    if is_eval:
                        # Optionally add just the key if you want to run blind
                        # file_list.append(key) 
                        pass
                    continue

                file_list.append(key)
                d_meta[key] = 1 if label == "bonafide" else 0
        except Exception:
            continue

    # Return the correct format based on the flags
    if is_train:
        return d_meta, file_list
    elif is_eval:
        return file_list
    else:
        # For dev set
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
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        try:
            # FIX: Use torchaudio.load to read file
            waveform, _ = torchaudio.load(str(self.base_dir / f"flac/{key}.flac"))
            X_numpy = waveform.numpy().squeeze()
        except Exception as e:
            print(f"Warning: Error loading {key}: {e}. Returning zeros.")
            X_numpy = np.zeros(self.cut) # Return dummy data

        X_pad = pad_random(X_numpy, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels.get(key, 0) # Use .get for safety
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        try:
            # FIX: Use torchaudio.load, just like in your training class
            waveform, _ = torchaudio.load(str(self.base_dir / f"flac/{key}.flac"))
            X_numpy = waveform.numpy().squeeze()
            X_pad = pad(X_numpy, self.cut) # Use the non-random pad
        except Exception as e:
            # If reading fails, create a silent audio clip instead of crashing
            print(f"\n[Warning] Failed to read {key}: {e}. Returning silent audio.\n")
            X_pad = np.zeros(self.cut, dtype=np.float32)
        
        x_inp = Tensor(X_pad)
        return x_inp, key
