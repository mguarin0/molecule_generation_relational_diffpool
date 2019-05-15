from math import ceil

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import Linear
#from torch_geometric.nn import GATConv, dense_diff_pool, RGCNConv
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

# TODO implement this when you find a way to incorporate multiple edge types into relational gcn
class Relational_GCN_Embedded(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(Relational_GCN, self).__init__()
        # 5 [128, 64] 5 0.0
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0]) # 5x128
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1]) # 128x64

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x9x5
        # adj : 16x4x9x9

        # equation 5 in molgan paper first layer
        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1) # 16x4x9x128
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden)) # 16x4x9x128
        hidden = torch.sum(hidden, 1) + self.linear1(input) # 16x9x128
        hidden = activation(hidden) if activation is not None else hidden # 16x9x128
        hidden = self.dropout(hidden)

        # equation 5 in molgan paper on next layer
        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1) # 16x4x9x64
        output = torch.einsum('bijk,bikl->bijl', (adj, output)) # 16x4x9x64
        output = torch.sum(output, 1) + self.linear2(hidden) # 16x9x64
        output = activation(output) if activation is not None else output # 16x9x64
        output = self.dropout(output)
        return output

class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)

        self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                   out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj))
        x2 = F.relu(self.conv2(x1, adj))
        return self.lin(torch.cat([x1, x2], dim=-1))


class DiffPool(torch.nn.Module):
    def __init__(self,
                 input_features_dim: int,
                 gcn_hidden_dims: list,
                 ff_hidden_dims: list,
                 nodes_dim: int,
                 num_classes: int,
                 layer_downsample_percents: list):
        super(DiffPool, self).__init__()

        num_nodes = ceil(layer_downsample_percents[0] * nodes_dim) # pool down
        self.embed_block1 = Block(input_features_dim, gcn_hidden_dims[0], gcn_hidden_dims[0])
        self.pool_block1 = Block(input_features_dim, gcn_hidden_dims[0], num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for downsample_percent, gcn_hidden_dim in list(zip(layer_downsample_percents[1:], gcn_hidden_dims[1:])):
            num_nodes = ceil(downsample_percent * num_nodes) # pool down
            self.embed_blocks.append(Block(gcn_hidden_dim, gcn_hidden_dim, gcn_hidden_dim))
            self.pool_blocks.append(Block(gcn_hidden_dim, gcn_hidden_dim, num_nodes))

        self.lin1_disc = Linear(gcn_hidden_dims[-1], ff_hidden_dims[0])
        self.lin2_disc = Linear(ff_hidden_dims[0], ff_hidden_dims[1])
        self.lin3_disc = Linear(ff_hidden_dims[1], num_classes)

        self.lin1_aux = Linear(gcn_hidden_dims[-1], ff_hidden_dims[0])
        self.lin2_aux = Linear(ff_hidden_dims[0], ff_hidden_dims[1])
        self.lin3_aux = Linear(ff_hidden_dims[1], num_classes)


    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for block1, block2 in zip(self.embed_blocks, self.pool_blocks):
            block1.reset_parameters()
            block2.reset_parameters()
        self.lin1_disc.reset_parameters()
        self.lin2_disc.reset_parameters()
        self.lin3_disc.reset_parameters()
        self.lin1_aux.reset_parameters()
        self.lin2_aux.reset_parameters()
        self.lin3_aux.reset_parameters()

    def forward(self, x, adj, mask):
        """
        x: [b, n, f]
        adj: [b, n, n]
        mask: [b, n]
        """
        s = self.pool_block1(x, adj, mask, add_loop=True) # [b, n, downsampled_dim]
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True)) # [b, n, downsampled_dim]
        xs = [x.mean(dim=1)]
        x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)

        for embed, pool in zip(self.embed_blocks, self.pool_blocks):
            s = pool(x, adj)
            x = F.relu(embed(x, adj))
            xs.append(x.mean(dim=1))
            x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)

        # disc
        x_disc = F.relu(self.lin1_disc(x))
        x_disc = F.dropout(x_disc, p=0.5, training=self.training)
        x_disc = F.relu(self.lin2_disc(x_disc))
        x_disc = F.dropout(x_disc, p=0.5, training=self.training)
        x_disc = self.lin3_disc(x_disc)
        x_disc = F.sigmoid(x_disc)

        # aux
        x_aux = F.relu(self.lin1_aux(x))
        x_aux = F.dropout(x_aux, p=0.5, training=self.training)
        x_aux = F.relu(self.lin2_aux(x_aux))
        x_aux = F.dropout(x_aux, p=0.5, training=self.training)
        x_aux = self.lin3_aux(x_aux)

        return x_disc, x_aux

    def __repr__(self):
        return self.__class__.__name__
