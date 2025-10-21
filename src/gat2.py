import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as conv
from torch_geometric.nn import PairNorm
import netlist

class GNN(torch.nn.Module):
    def __init__(self, features, classes, width, depth, activation, dropout, pairnorm, heads):
        super().__init__()
        self.depth = depth
        self.width = width
        self.heads = heads

        self.layers = torch.nn.ModuleList([conv(features, width, heads=heads)])
        for i in range(depth - 2):
            self.layers.append(conv(-1, width, heads=heads))
        self.layers.append(conv(-1, classes, heads=heads, concat=False))

        self.act = activation
        self.dropout = torch.nn.Identity()
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.pairnorm = torch.nn.Identity()
        if pairnorm:
            self.pairnorm = PairNorm(scale=pairnorm, scale_individually=True)

    def forward(self, x, edge_index):
        for l in self.layers[:-1]:
            x = l(x, edge_index)
            x = self.pairnorm(x)
            x = self.act(x)

            # This is technically the dropout for the "next" layer, but putting it here avoids an extra "if" statement.
            # See https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
            # We don't want dropout on either the inputs or the output logits.
            x = self.dropout(x)

        x = self.layers[-1](x, edge_index)
        return x

def rehydrate(classes, width, depth, act, weights, pairnorm, heads):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")
    gnn = GNN(netlist.features, len(classes), width, depth, act, None, 95 if pairnorm is None else pairnorm, heads=heads)
    gnn.load_state_dict(weights)
    return gnn
