from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
#from torch_geometric.nn import GATConv, dense_diff_pool, RGCNConv
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


# @author of this class is yongqyu
class RGCN_Block(nn.Module):

  def __init__(self, x_dim, layer_dims, dropout):
    super(RGCN_Block, self).__init__()

    # set up rgcn layers
    layer_dims = [x_dim] + layer_dims
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
      print(layer)
      hi = layer(x)
      out = torch.stack([layer(x) for _ in range(adj.size(1))], 1)
      print(out.size())
      out = hi+torch.sum(torch.einsum('brnv,brve->brne', (adj, out)), 1)
      print(out.size())
      out = activation(out) if activation is not None else out
      print(out.size())
      out = self.dropout(out)
      print(out.size())
      x=out
    return out # bxnxe


class DiffPool_Block(nn.Module):
  def __init__(self, layer_dims):
    super(DiffPool_Block, self).__init__()

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
               n_dim,
               z_dim,
               embed_rgcn_layer_params,
               pool_rgcn_layer_params,
               embed_block_layer_params,
               pool_block_layer_params,
               ff_layer_params):
    super(DiffPool, self).__init__()

    self.embed_blocks = nn.ModuleList()
    self.pool_blocks = nn.ModuleList()
    self.embed_blocks.append(RGCN_Block(x_dim,
                                        embed_rgcn_layer_params[0],
                                        embed_rgcn_layer_params[1]))
    self.pool_blocks.append(RGCN_Block(x_dim,
                                       self.get_lnn_layer_dim(n_dim, pool_rgcn_layer_params[0]),
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

  def get_lnn_layer_dim(self, n, pool_percents):
    pool_dims = []
    for pool_percent in pool_percents:
      n = ceil(n * pool_percent)
      pool_dims.append(n)
    return pool_dims 

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

  def forward(self, x, adj, mask=None):
    """
    x: [b, n, f]
    adj: [b, n, n]
    mask: [b, n]
    """
    print("x: {}".format(x.shape))
    print("adj: {}".format(adj.shape))

    for embed, pool in zip(self.embed_blocks, self.pool_blocks):
      s = pool(x, adj)
      x = embed(x, adj)
      print("first pass")
      print("x: {}".format(x.shape))
      print("s: {}".format(s.shape))
      exit(0)
