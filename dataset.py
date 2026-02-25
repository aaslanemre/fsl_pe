import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class MPIIFewShotDataset(Dataset):
    def __init__(self, json_path, img_dir, n_shot=5, target_size=8):
        self.img_dir = img_dir
        self.n_shot = n_shot
        self.target_size = target_size # The 8x8 output from our model
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def load_item(self, idx):
        sample = self.data[idx]
        path = os.path.join(self.img_dir, sample['image'])
        img = cv2.imread(path)
        if img is None: # Safety check
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        h, w = img.shape[:2]
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Scale joints to 256x256
        joints = np.array(sample['joints'])
        joints[:, 0] = joints[:, 0] * (256.0 / w)
        joints[:, 1] = joints[:, 1] * (256.0 / h)
        
        return img_tensor, joints

    def generate_heatmap(self, joints):
        # Creates 16 heatmaps of size 8x8
        target = np.zeros((16, self.target_size, self.target_size), dtype=np.float32)
        sigma = 1.0
        for i in range(16):
            # Scale 256 coord to 8
            mu_x = joints[i][0] * (self.target_size / 256.0)
            mu_y = joints[i][1] * (self.target_size / 256.0)
            
            for y in range(self.target_size):
                for x in range(self.target_size):
                    dist_sq = (x - mu_x)**2 + (y - mu_y)**2
                    target[i, y, x] = np.exp(-dist_sq / (2 * sigma**2))
        return torch.from_numpy(target)

    def __getitem__(self, idx):
        # Support Set
        support_indices = np.random.choice(len(self.data), self.n_shot, replace=False)
        s_list = [self.load_item(i)[0] for i in support_indices]
        
        # Query Set
        q_img, q_joints = self.load_item(idx)
        target = self.generate_heatmap(q_joints)
        
        return {
            "support": torch.stack(s_list), 
            "query": q_img,
            "target": target
        }
