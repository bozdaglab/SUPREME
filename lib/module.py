import os
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from collections import Counter

class Net(torch.nn.Module):
    def __init__(self, in_size=16, hid_size=8, out_size=2):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb


