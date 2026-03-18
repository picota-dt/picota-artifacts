import torch
from torch.utils.data import Dataset

import Device


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.device = Device.get_device()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'out': torch.tensor(item['out'], dtype=torch.float32).to(device=self.device),
            't': torch.tensor(item['t'], dtype=torch.float32).to(device=self.device),
            'categorical_t_features': torch.tensor(item['categorical_t_features'], dtype=torch.float32).to(
                device=self.device),
            'numerical_t_features': torch.tensor(item['numerical_t_features'], dtype=torch.float32).to(
                device=self.device),
            'lookback_t': torch.tensor(item['lookback_t'], dtype=torch.float32).to(device=self.device),
            'categorical_lookback_features': torch.tensor(item['categorical_lookback_features'],
                                                          dtype=torch.float32).to(device=self.device),
            'numerical_lookback_features': torch.tensor(item['numerical_lookback_features'], dtype=torch.float32).to(
                device=self.device)
        }
