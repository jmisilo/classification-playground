import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, one_hot=None):
        self.data = data.astype(np.float32) # map to float32 instead of float64
        self.labels = labels

        self.one_hot = np.unique(self.labels).size > 2 if one_hot is None else one_hot

        self.num_features = self.data.shape[-1]
        self.num_labels = self.labels.max() + 1 if np.unique(self.labels).size > 2 else 1

        # if self.one_hot:
            # self._one_hot()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index] if self.one_hot else np.expand_dims(self.labels[index], axis=-1)