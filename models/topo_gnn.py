import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge

from layers.rephine_layer import RephineLayer
from models.gnn import GNN


class TopoGNN(GNN):
    def __init__(
        self,
        hidden_dim,
        depth,
        num_node_features,
        num_classes,
        gnn,
        num_filtrations,
        filtration_hidden,
        out_ph_dim,
        diagram_type="rephine",
        ph_pooling_type="mean",
        dim1=True,
        sig_filtrations=True,
        global_pooling="mean",
        deg=None,
        batch_norm=False,
    ):
        super().__init__(
            gnn=gnn,
            hidden_dim=hidden_dim,
            depth=depth,
            num_node_features=num_node_features,
            num_classes=num_classes,
            deg=deg,
            global_pooling=global_pooling,
            batch_norm=batch_norm,
        )

        topo_layers = []
        self.ph_pooling_type = ph_pooling_type
        for i in range(len(self.layers)):
            topo = RephineLayer(
                n_features=hidden_dim,
                n_filtrations=num_filtrations,
                filtration_hidden=filtration_hidden,
                out_dim=out_ph_dim,
                diagram_type=diagram_type,
                dim1=dim1,
                sig_filtrations=sig_filtrations,
            )
            topo_layers.append(topo)

        self.ph_layers = nn.ModuleList(topo_layers)

        final_dim = (
            hidden_dim + len(self.ph_layers) * out_ph_dim
            if self.ph_pooling_type == "cat"
            else hidden_dim + out_ph_dim
        )
        self.classif = torch.nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        if self.ph_pooling_type != "mean":
            self.jump = JumpingKnowledge(mode=self.ph_pooling_type)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x.float())

        ph_vectors = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index)
            ph_vectors += [self.ph_layers[i](x, data)]

        # Pooling GNN embeddings
        x = self.pooling_fun(x, data.batch)

        # Pooling PH diagrams
        if self.ph_pooling_type == "mean":
            ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        else:
            ph_embedding = self.jump(ph_vectors)
        x_pre_class = torch.cat([x, ph_embedding], axis=1)

        # Final classification
        x = self.classif(x_pre_class)
        return x
