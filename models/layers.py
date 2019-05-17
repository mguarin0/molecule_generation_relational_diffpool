from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
#from torch_geometric.nn import GATConv, dense_diff_pool, RGCNConv
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


# @author of this class is yongqyu
class RGCN(nn.Module):

  def __init__(self, layer_dims, dropout):
    super(RGCN, self).__init__()

    # set up rgcn layers
    self.layers = nn.ModuleList([nn.Linear(ln, lnn)
                                   for ln, lnn in zip(layer_dims[:-1],
                                                      layer_dims[1:])])
    self.dropout = nn.Dropout(dropout)

  def reset_parameters(self):
    for layer in self.layers:
      layer.reset_parameters()

  def forward(self, x, adj, activation=None):
    # input : bxnxd
    # adj : bxrxnxn

    for layer in self.layers:
      out = torch.stack([layer(x) for _ in range(adj.size(1))], 1)
      out = torch.sum(torch.einsum('brnv,brve->brne', (adj, out)), 1) + layer(x)
      out = activation(out) if activation is not None else out
      out = self.dropout(out)
    return out # bxnxe


class DiffPool_Block(nn.Module):
  def __init__(self, layer_dims):
    super(Block, self).__init__()

    # set up gcn layers
    self.layers = nn.ModuleList([DenseSAGEConv(ln, lnn)
                                   for ln, lnn in zip(layer_dims[:-1],
                                                      layer_dims[1:])])
    # set up ff layers
    self.layers.append(torch.nn.Linear(layer_dims[-2] + layer_dims[-1],
                                       layer_dims[-1]))

  def reset_parameters(self):
    for layer in self.layers:
      layer.reset_parameters()

  def forward(self, x, adj, mask=None, add_loop=True):
    xs = []
    for layer in self.layers[:-1]:
      x = F.relu(layer(x, adj))
      xs.append(x)
    return self.layers[-1](torch.cat([xs[-2],
                                      xs[-1]], dim=-1))


class DiffPool(nn.Module):
  def __init__(self,
               x_dim,
               r_dim,
               z_dim,
               embed_rgcn_layer_params,
               pool_rgcn_layer_params,
               embed_block_layer_params,
               pool_block_layer_params,
               ff_layer_params):
    super(DiffPool, self).__init__()

    self.node_dim = x_dim
    self.embed = nn.ModuleList()
    self.pool = nn.ModuleList()
    self.embed.append(RGCN(embed_rgcn_layer_params[0], embed_rgcn_layer_params[1]))
    self.pool.append(RGCN([self.get_lnn_layer_dim(pool_percnt)
                        for pool_percnt in pool_rgcn_layer_params[0]],
                     pool_rgcn_layer_params[1]))
    """
    num_nodes = ceil(layer_downsample_percents[0] * nodes_dim) # pool down
    self.embed_block1 = DiffPool_Block(input_x_dim, gcn_hidden_dims[0], gcn_hidden_dims[0])
    self.pool_block1 = DiffPool_Block(input_x_dim, gcn_hidden_dims[0], num_nodes)

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
    """

  def get_lnn_layer_dim(pool_percent):
    self.node_dim = ceil(self.node_dim * pool_percent)
    return self.node_dim


  """
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
  """

  def forward(self, x, adj, mask):
    print("forward")
    """
    x: [b, n, f]
    adj: [b, n, n]
    mask: [b, n]
    """
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
    """
