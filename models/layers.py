from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


# @author of this class is yongqyu
class RGCN_Block(nn.Module):

  def __init__(self, d_dim, layer_dims, dropout):
    super(RGCN_Block, self).__init__()

    layer_dims = [d_dim] + layer_dims
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
  def __init__(self, d_dim, hidden_dim, out_dim, dropout):
    super(GCN_Block, self).__init__()

    # set up gcn layers
    self.gcn_layers = nn.ModuleList([DenseSAGEConv(d_dim, hidden_dim),
                                     DenseSAGEConv(hidden_dim, out_dim)])
    # set up ff layers
    self.ff_layer = Linear(hidden_dim+out_dim, out_dim)

  def reset_parameters(self):
    for layer in self.layers:
      layer.reset_parameters()

  def forward(self, x, adj, activation=None, mask=None, add_loop=True):
    xs = []
    for gcn_layer in self.gcn_layers:
      x = gcn_layer(x, adj)
      x = activation(x) if activation is not None else x 
      xs.append(x)
    return self.ff_layer(torch.cat([xs[-2],
                                    xs[-1]], dim=-1))

class DiffPool(nn.Module):
  def __init__(self,
               d_dim,
               r_dim,
               n_dim,
               z_dim,
               num_classes,
               embed_rgcn_layer_params,
               pool_rgcn_layer_params,
               embed_gcn_layer_params,
               pool_gcn_layer_params,
               ff_layer_params,
               module_type):
    super(DiffPool, self).__init__()

    self.module_type = module_type
    self.d_dim = d_dim
    self.r_dim = r_dim
    self.n_dim = n_dim
    self.z_dim = z_dim
    self.num_classes = num_classes 

    # TODO make this more dynamic low priority
    if module_type == "discriminator":
      rgcn_pooled_layer_dims = self.get_pooled_layer_dims(n_dim, pool_rgcn_layer_params[0])
      gcn_pooled_layer_dims = self.get_pooled_layer_dims(n_dim, pool_gcn_layer_params[0])

      self.embed_blocks = nn.ModuleDict([
        ["rgcn", RGCN_Block(d_dim,
                              embed_rgcn_layer_params[0],
                              embed_rgcn_layer_params[1])],
        ["gsage_l", GCN_Block(embed_rgcn_layer_params[0][-1],
                              embed_gcn_layer_params[0][0],
                              embed_gcn_layer_params[0][0],
                              embed_gcn_layer_params[1])],
        ["gsage_ll", GCN_Block(embed_gcn_layer_params[0][0],
                                 embed_gcn_layer_params[0][0],
                                 embed_gcn_layer_params[0][1],
                                 embed_gcn_layer_params[1])]
        ])

      self.pool_blocks = nn.ModuleDict([
        ["rgcn", RGCN_Block(d_dim,
                           rgcn_pooled_layer_dims,
                           pool_rgcn_layer_params[1])],
        ["gsage_l", GCN_Block(embed_rgcn_layer_params[0][-1],
                              embed_gcn_layer_params[0][0],
                              gcn_pooled_layer_dims[0],
                              pool_gcn_layer_params[1])],
        ["gsage_ll", GCN_Block(embed_gcn_layer_params[0][0],
                              embed_gcn_layer_params[0][0],
                              gcn_pooled_layer_dims[1],
                              pool_gcn_layer_params[1])]
        ])

      self.fc_graph_agg = nn.Sequential(Linear(gcn_pooled_layer_dims[1] * embed_gcn_layer_params[0][1],
                                               ff_layer_params[0][0]),
                                        nn.Tanh(),
                                        nn.Dropout(p=ff_layer_params[1], inplace=True),
                                        Linear(ff_layer_params[0][0],
                                               ff_layer_params[0][1]),
                                        nn.Tanh(),
                                        nn.Dropout(p=ff_layer_params[1], inplace=True))
      self.disc = Linear(ff_layer_params[0][1], self.num_classes)
      
    elif module_type == "generator":
      self.embed_blocks = nn.ModuleDict([
        ["rgcn", RGCN_Block(d_dim,
                            embed_rgcn_layer_params[0]+[d_dim],
                            embed_rgcn_layer_params[1])]
        ])

      self.decode_z_layers = []
      for ln, lnn in zip([z_dim]+ff_layer_params[0][:-1], ff_layer_params[0]):
        self.decode_z_layers.append(nn.Linear(ln, lnn))
        self.decode_z_layers.append(nn.Tanh())
        self.decode_z_layers.append(nn.Dropout(p=ff_layer_params[1], inplace=True))
      self.decode_z_layer = nn.Sequential(*self.decode_z_layers)
      self.decoder_dropout = nn.Dropout(p=ff_layer_params[1])
      self.x_layer = Linear(ff_layer_params[0][-1], d_dim*n_dim)
      self.adj_layer = Linear(ff_layer_params[0][-1], (n_dim**2)*r_dim)
#     self.final_adj_layer = nn.Sequential(Linear((n_dim**2)*r_dim, 512),
#                                          nn.ReLU(),
#                                          nn.Dropout(p=0.5, inplace=True),
#                                          Linear(512, (n_dim**2)*r_dim))
#     self.final_x_layer = nn.Sequential(Linear(n_dim*d_dim, 128),
#                                        nn.ReLU(),
#                                        nn.Dropout(p=0.5, inplace=True),
#                                        Linear(128, n_dim*d_dim))


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
    if self.module_type == "discriminator":
      x, adj, rel_adj = input
      for i, (k_embed, k_pool) in enumerate(zip(self.embed_blocks, self.pool_blocks)):
        if k_embed=="rgcn" and k_pool=="rgcn":
          s = self.pool_blocks[k_pool](x, rel_adj)
          x = self.embed_blocks[k_embed](x, rel_adj)
        elif "_".join(k_embed.split("_")[:-1])=="gsage" and "_".join(k_pool.split("_")[:-1])=="gsage":
          s = self.pool_blocks[k_pool](x, adj)
          x = self.embed_blocks[k_embed](x, adj)
        x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)
      out = self.fc_graph_agg(torch.reshape(x, (x.size(0), -1)))
      return out, link_loss

    elif self.module_type == "generator":
      z = input
      out = self.decode_z_layer(z)
      x_logits  = self.x_layer(out)\
                    .view(-1,
                          self.n_dim,
                          self.d_dim)
      adj_logits = self.adj_layer(out)\
                     .view(-1,
                           self.r_dim,
                           self.n_dim,
                           self.n_dim)
#     for k_embed in self.embed_blocks:
#       if "_".join(k_embed.split("_")[:-1])=="rgcn":
#         x = self.embed_blocks[k_embed](x, rel_adj)
#     adj_logits = self.final_adj_layer(torch.reshape(adj_logits,
#                                                     (adj_logits.size(0), -1)))\
#                    .view(-1, self.r_dim,
#                          self.n_dim, self.n_dim)
      adj_logits = (adj_logits + adj_logits.permute(0,1,3,2))/2 # avg permutation nodes 
      adj_logits = self.decoder_dropout(adj_logits.permute(0,2,3,1))
#     x_logits = self.final_x_layer(torch.reshape(x, (x.size(0), -1)))\
#                  .view(-1, self.n_dim,
#                          self.d_dim)
      x_logits = self.decoder_dropout(x_logits)
      adj = F.softmax(adj_logits/1.0, -1)
      x = F.softmax(x_logits/1.0, -1)
      return x, adj
