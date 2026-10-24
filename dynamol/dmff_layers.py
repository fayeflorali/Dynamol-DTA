import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool


class SpatialGroupEnhance_for_1D(nn.Module):
    def __init__(self, groups=32):
        super(SpatialGroupEnhance_for_1D, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1))
        self.bias = Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch, channels, length = x.size()
        x = x.view(batch * self.groups, -1, length)
        scaled = x * self.avg_pool(x)
        scaled = scaled.sum(dim=1, keepdim=True)
        normalized = scaled.view(batch * self.groups, -1)
        normalized = normalized - normalized.mean(dim=1, keepdim=True)
        normalized = normalized / (normalized.std(dim=1, keepdim=True) + 1e-5)
        normalized = normalized.view(batch, self.groups, length)
        normalized = normalized * self.weight + self.bias
        normalized = normalized.view(batch * self.groups, 1, length)
        x = x * self.sig(normalized)
        return x.view(batch, channels, length)


class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)
        minus_inf = -9e15 * torch.ones_like(query)
        weights = self.softmax(torch.where(masks > 0.5, query, minus_inf))
        output = torch.matmul(weights, x).mean(dim=1)
        return output, weights


class GINConvNet(nn.Module):
    """Original DMFF five-layer GIN.

    Layer structure, operator order and parameter names follow
    ``DMFF-DTA-main/model.py`` exactly: each block applies ReLU to the GIN
    output first and BatchNorm second, and the submodules are named
    ``conv1..conv5`` / ``bn1..bn5``. Keeping the names identical means
    ``state_dict`` keys stay interchangeable with the reference implementation,
    so a DMFF-DTA-main checkpoint loads without remapping.

    Only the ``forward`` signature differs: it takes ``(x, edge_index, batch)``
    instead of a PyG ``Data`` object, because the sparse pocket modes assemble
    the node block online inside ``DMFF.forward`` and never materialise
    ``Data.x``. The reference implementation reads the same three tensors off
    ``data`` in its first line, so the computation is unchanged.
    """

    def __init__(self, n_output=1, num_features_xd=128, output_dim=128,
                 dropout=0.2):
        super(GINConvNet, self).__init__()
        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.out(x)
