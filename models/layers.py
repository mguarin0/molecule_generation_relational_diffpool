from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


# @author of this class is yongqyu
class RGCN_Block(nn.Module):

  def __init__(self, x_dim, layer_dims, dropout):
    super(RGCN_Block, self).__init__()

    layer_dims = [x_dim] + layer_dims
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
      hi = layer(x)
      out = torch.stack([layer(x) for _ in range(adj.size(1))], 1)
      out = hi+torch.sum(torch.einsum('brnv,brve->brne', (adj, out)), 1)
      out = activation(out) if activation is not None else out
      out = self.dropout(out)
      x=out
    return out # bxnxe


class GCN_Block(nn.Module):
  def __init__(self, x_dim, hidden_dim, out_dim, dropout):
    super(GCN_Block, self).__init__()

    # set up gcn layers
    self.gcn_layers = nn.ModuleList([DenseSAGEConv(x_dim, hidden_dim),
                                     DenseSAGEConv(hidden_dim, out_dim)])
    # set up ff layers
    self.ff_layer = Linear(hidden_dim+out_dim,
                           out_dim)
    print("========")
    print("gcn_layers")
    for layer in self.gcn_layers:
      print(layer)
    print("ff_layers")
    print(self.ff_layer)

  def reset_parameters(self):
    for layer in self.layers:
      layer.reset_parameters()

  def forward(self, x, adj, mask=None, add_loop=True):
    xs = []
    for gcn_layer in self.gcn_layers:
      x = F.relu(gcn_layer(x, adj))
      xs.append(x)
    return self.ff_layer(torch.cat([xs[-2],
                                    xs[-1]], dim=-1))

class DiffPool(nn.Module):
  def __init__(self,
               x_dim,
               r_dim,
               n_dim,
               z_dim,
               embed_rgcn_layer_params,
               pool_rgcn_layer_params,
               embed_gcn_layer_params,
               pool_gcn_layer_params,
               ff_layer_params,
               module_type):
    super(DiffPool, self).__init__()

    self.module_type = module_type
    self.embed_gcn_layer_params = embed_gcn_layer_params
    # TODO make this more dynamic low priority
    self.embed_blocks = nn.ModuleDict([
      ["rgcn", RGCN_Block(x_dim,
                         embed_rgcn_layer_params[0],
                         embed_rgcn_layer_params[1])],
      ["gsage_1", GCN_Block(embed_rgcn_layer_params[0][-1],
                            embed_gcn_layer_params[0][0],
                            embed_gcn_layer_params[0][0],
                            embed_gcn_layer_params[1])],
      ["gsage_2", GCN_Block(embed_gcn_layer_params[0][0],
                            embed_gcn_layer_params[0][0],
                            embed_gcn_layer_params[0][1],
                            embed_gcn_layer_params[1])]
      ])
    self.rgcn_pooled_layer_dims = self.get_pooled_layer_dims(n_dim, pool_rgcn_layer_params[0])
    self.gcn_pooled_layer_dims = self.get_pooled_layer_dims(n_dim, pool_gcn_layer_params[0])
    self.pool_blocks = nn.ModuleDict([
      ["rgcn", RGCN_Block(x_dim,
                         self.rgcn_pooled_layer_dims,
                         pool_rgcn_layer_params[1])],
      ["gsage_1", GCN_Block(embed_rgcn_layer_params[0][-1],
                            embed_gcn_layer_params[0][0],
                            self.gcn_pooled_layer_dims[0],
                            pool_gcn_layer_params[1])],
      ["gsage_2", GCN_Block(embed_gcn_layer_params[0][0],
                            embed_gcn_layer_params[0][0],
                            self.gcn_pooled_layer_dims[1],
                            pool_gcn_layer_params[1])]
      ])

    if module_type == "encoder":
      # TODO regularize
      self.encoder_z_layer = nn.Sequential(Linear(self.gcn_pooled_layer_dims[1] * embed_gcn_layer_params[0][1],
                                       ff_layer_params[0][0]),
                                nn.Tanh(),
                                Linear(ff_layer_params[0][0],
                                       z_dim))
    elif module_type == "decoder":
      self.decode_z_layers = []
      for ln, lnn in zip([z_dim]+ff_layer_params[0][:-1], ff_layer_params[0]):
        self.decode_z_layers.append(nn.Linear(ln, lnn))
        self.decode_z_layers.append(nn.Tanh())
        self.decode_z_layers.append(nn.Dropout(p=ff_layer_params[1],
                                                    inplace=True))
      self.decode_z_layer = nn.Sequential(*self.decode_z_layers)
      self.x_layer = Linear(ff_layer_params[0][-1],
                             self.gcn_pooled_layer_dims[0] * embed_gcn_layer_params[0][0])
      self.adj_layer = Linear(ff_layer_params[0][-1],
                               self.gcn_pooled_layer_dims[0]**2)
      self.decoder_dropout = nn.Dropout(p=ff_layer_params[1])


  def get_pooled_layer_dims(self, n, pool_percents):
    pool_dims = []
    for pool_percent in pool_percents:
      n = ceil(n * pool_percent)
      pool_dims.append(n)
    if self.module_type == "decoder":
      pool_dims = list(reversed(pool_dims))
    return pool_dims 

  def forward(self, input):
    """
    x: [b, n, f]
    adj: [b, n, n]
    mask: [b, n]
    """
    if self.module_type == "encoder":
       x, adj, rel_adj = input
    elif self.module_type == "decoder":
       z = input
       out = self.decode_z_layer(z)
       x_logits = self.x_layer(out)\
                    .view(-1,
                          self.gcn_pooled_layer_dims[0],
                          self.embed_gcn_layer_params[0][0])
       adj_logits = self.adj_layer(out)\
                      .view(-1,
                            self.gcn_pooled_layer_dims[0],
                            self.gcn_pooled_layer_dims[0])
       x = x_logits.tanh()
       adj = F.softmax(adj_logits, -1) # should have effect of categorical sampling

    for i, (k_embed, k_pool) in enumerate(zip(self.embed_blocks, self.pool_blocks)):
      print(i, k_embed, k_pool)
      if k_embed=="rgcn" and k_pool=="rgcn":
        s = self.pool_blocks[k_pool](x, rel_adj)
        x = self.embed_blocks[k_embed](x, rel_adj)
#      elif k_embed=="gsage" and k_pool=="gsage":
      else:
        s = self.pool_blocks[k_pool](x, adj)
        x = self.embed_blocks[k_embed](x, adj)
      x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)
      print("x: {}".format(x.shape))
      print("adj: {}".format(adj.shape))

    if self.module_type == "encoder":
      z = self.encoder_z_layer(torch.reshape(x, (x.size(0), -1)))
      return z
