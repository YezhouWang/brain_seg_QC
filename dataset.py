import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os

class BrainSegQCDataset(Dataset):
    def __init__(self, t1_dir, seg_init_dir, seg_qc_dir):
        self.t1_paths = sorted([os.path.join(t1_dir, f) for f in os.listdir(t1_dir)])
        self.init_seg_paths = sorted([os.path.join(seg_init_dir, f) for f in os.listdir(seg_init_dir)])
        self.qc_seg_paths = sorted([os.path.join(seg_qc_dir, f) for f in os.listdir(seg_qc_dir)])

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, idx):
        t1 = nib.load(self.t1_paths[idx]).get_fdata()
        seg_init = nib.load(self.init_seg_paths[idx]).get_fdata()
        seg_qc = nib.load(self.qc_seg_paths[idx]).get_fdata()

        x = np.stack([t1, seg_init], axis=0)
        y = seg_qc

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
      
