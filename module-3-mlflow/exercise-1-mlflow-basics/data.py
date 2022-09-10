import json
import numpy as np
from dataclasses import dataclass

@dataclass
class Dataset:
    """
    Simple container for holding train, validation, and test data.
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    
    @property
    def X_all(self):
        return np.concatenate([self.X_train, self.X_val, self.X_test], axis=0)
    
    @property
    def y_all(self):
        return np.concatenate([self.y_train, self.y_val, self.y_test], axis=0)
    
    def __repr__(self):
        msg = "Dataset: {} features, {} train, {} val, {} test samples"
        return msg.format(self.X_train.shape[1], 
                          len(self.y_train), len(self.y_val), len(self.y_test))


def load_data(file='data.json', seed=None):
    """
    Load data into a Dataset object.
    
    Parameters
    ----------
    file: string or Path object
        Path to file to load data from, default 'data.json'.
    seed: int or None
        Random seed used to shuffle data into train/validation/test sets. If None,
        a new seed will be randomly selected.
    
    Returns
    ---------
    data: Dataset object
        Contains loaded data.
    """
    with open(file, 'r') as f:
        data = json.load(f)
    rng = np.random.default_rng(seed)
    features = np.stack(data['features'])
    labels = np.array(data['labels'])
    
    X_data = [[] for _ in range(3)]
    
    y_data = [[] for _ in range(3)]
    
    for i in range(6):
        subset = features[labels == i]
        n = len(subset)
        idx = [0, int(0.7*n), int(0.85*n), n]
        rng.shuffle(subset)
        
        for j, (n1, n2) in enumerate(zip(idx[:-1], idx[1:])):
            xdata = subset[n1:n2]
            X_data[j].append(xdata)
            y_data[j].append(np.zeros(len(xdata), np.uint8) + i)
    
    for i, (data, labels_y) in enumerate(zip(X_data, y_data)):
        Xdata = np.concatenate(data, axis=0)
        ydata = np.concatenate(labels_y, axis=0)
        idx = np.arange(len(ydata))
        rng.shuffle(idx)
        X_data[i] = Xdata[idx]
        y_data[i] = ydata[idx]
        
    
    dataset = Dataset(*X_data, *y_data)
    return dataset