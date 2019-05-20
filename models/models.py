import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

class DiPol_Gen(nn.Module):
  def __init__(self, d_dim, r_dim, n_dim, z_dim, num_classes, model_config):
    super(DiPol_Gen, self).__init__()

    self.gen = DiffPool(d_dim,
                       r_dim,
                       n_dim,
                       z_dim,
                       num_classes,
                       model_config["gen"]["embed_rgcn_layer_params"],
                       model_config["gen"]["pool_rgcn_layer_params"],
                       model_config["gen"]["embed_gcn_layer_params"],
                       model_config["gen"]["pool_gcn_layer_params"],
                       model_config["gen"]["ff_layer_params"],
                       "gen")


  def forward(self, input, module_type="gen"):
    x_logits, rel_adj_logits = self.gen[module_type](input)
    rel_adj = F.softmax(rel_adj_logits/1.0, -1)
    x = F.softmax(x_logits/1.0, -1)
    # TODO gumbel softmax


class DiPol_Dscr(nn.Module):
  def __init__(self, d_dim, r_dim, n_dim, z_dim, num_classes, model_config):
    super(DiPol_Dscr, self).__init__()
    self.dscr = DiffPool(d_dim,
                         r_dim,
                         n_dim,
                         z_dim,
                         num_classes,
                         model_config["dscr"]["embed_rgcn_layer_params"],
                         model_config["dscr"]["pool_rgcn_layer_params"],
                         model_config["dscr"]["embed_gcn_layer_params"],
                         model_config["dscr"]["pool_gcn_layer_params"],
                         model_config["dscr"]["ff_layer_params"],
                         "dscr")],
  def forward(self, input, module_type="dscr"):
    pred_logits, link_losses, ent_losses = self.dscr[module_type](input)
    F.sigmoid(pred_logits)
