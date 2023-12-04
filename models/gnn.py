import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_add_pool, global_mean_pool, GCNConv

from layers.gin_layer import GINLayer
from layers.graph_convolution_layer import GCNLayer
from models.gnn_factory.gcn_layer_factory import GcnCreator
from models.gnn_factory.gin_layer_factory import GinCreator
from models.gnn_factory.pna_layer_factory import PnaCreator


class GNN(nn.Module):
    def __init__(
        self,
        gnn,
        hidden_dim,
        depth,
        num_node_features,
        num_classes,
        global_pooling,
        deg=None,
        batch_norm=True,
    ):
        super().__init__()
        if gnn == "gin":
            gnn_instance = GinCreator(hidden_dim, batch_norm)
        elif gnn == "gcn":
            gnn_instance = GcnCreator(hidden_dim, batch_norm)

        build_gnn_layer = gnn_instance.return_gnn_instance
        if global_pooling == "mean":
            graph_pooling_operation = global_mean_pool
        elif global_pooling == "sum":
            graph_pooling_operation = global_add_pool

        self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        layers = [build_gnn_layer(is_last=i == (depth - 1)) for i in range(depth)]

        self.layers = nn.ModuleList(layers)

        dim_before_class = hidden_dim
        self.classif = torch.nn.Sequential(
            nn.Linear(dim_before_class, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x.float())

        for layer in self.layers:
            x = layer(x, edge_index=edge_index)

        x = self.pooling_fun(x, data.batch)
        x = self.classif(x)
        return x


# Used in the toy example
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 36)
        self.conv2 = GCNConv(36, 16)
        self.out = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 1))
        self.bn = nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)
        h = x.clone().detach()
        x = self.bn(x)
        x = self.out(x)
        return x, h