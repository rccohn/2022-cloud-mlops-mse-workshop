import json
import torch
from torch_geometric.data import Data


def load_data(path, thresh=10):
    with open(path, 'r') as f:
        json_data = json.load(f)
    data = Data()
    for key in ('x', 'edge_index', 'y'):
        data[key] = torch.Tensor(json_data[key])
    data['x'] = data['x'].double()
    data['edge_index'] = data['edge_index'].long()
    data['candidate_mask'] = data['x'][:, 0] == 1.
    data['y'] = (data['y'] > thresh).long()
    return data
