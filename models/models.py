import torch.nn as nn
from models.layers import *

class DiPol_GraphVAE(nn.Module):
  def __init__(self, x_dim, r_dim, n_dim, z_dim, model_config):
    super(DiPol_GraphVAE, self).__init__()

    # encoder definition
    self.dipol_modules = nn.ModuleDict([
      ["encoder", DiffPool(x_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          model_config["encoder"]["embed_rgcn_layer_params"],
                          model_config["encoder"]["pool_rgcn_layer_params"],
                          model_config["encoder"]["embed_gcn_layer_params"],
                          model_config["encoder"]["pool_gcn_layer_params"],
                          model_config["encoder"]["ff_layer_params"],
                          "encoder")],
      ["decoder", DiffPool(x_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          model_config["decoder"]["embed_rgcn_layer_params"],
                          model_config["decoder"]["pool_rgcn_layer_params"],
                          model_config["decoder"]["embed_gcn_layer_params"],
                          model_config["decoder"]["pool_gcn_layer_params"],
                          model_config["decoder"]["ff_layer_params"],
                          "decoder")]
      ])

  def forward(self, input, module_type):
    if module_type == "encoder":
      x, adj, rel_adj = input
      return self.dipol_modules[module_type]((x, adj, rel_adj))
    elif module_type == "decoder":
      z = input
      return self.dipol_modules[module_type]((z))

