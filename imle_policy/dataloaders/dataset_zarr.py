import torch
from torch.utils.data import Dataset 
import zarr 
import numpy as np

class PolicyDataset(Dataset):
    def __init__(self, dataset_path, pred_horizon=1, obs_horizon=1, action_horizon=1, dataset_percentage=1.0):
        self.states = zarr.open(f"{dataset_path}/state", mode="r")
        self.actions = zarr.open(f"{dataset_path}/action", mode="r")
        
        N = int(self.states.shape[0] * dataset_percentage)
        self.states = self.states[:N]
        self.actions = self.actions[:N]
        
        self.stats = {
            "state_mean": np.mean(self.states, axis=0),
            "state_std": np.std(self.states, axis=0),
            "action_mean": np.mean(self.actions, axis=0),
            "action_std": np.std(self.actions, axis=0),
        }
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        
    def __len__(self):
        return self.states.shape[0] - self.obs_horizon - self.pred_horizon 
    
    def __getitem__(self, idx):
        obs = self.states[idx : idx + self.obs_horizon]
        action = self.actions[idx + self.obs_horizon : idx + self.obs_horizon + self.pred_horizon]
        
        return {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
        }