import torch.nn as nn
from models.layers import *

class DiPol_GraphVAE(nn.Module):
  def __init__(self, x_dim, r_dim, n_dim, z_dim, model_config):
    super(DiPol_GraphVAE, self).__init__()

    # encoder definition
    self.dipol_modules = nn.ModuleDict({
      "encoder": DiffPool(x_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          model_config["encoder"]["embed_rgcn_layer_params"],
                          model_config["encoder"]["pool_rgcn_layer_params"],
                          model_config["encoder"]["embed_block_layer_params"],
                          model_config["encoder"]["pool_block_layer_params"],
                          model_config["encoder"]["ff_layer_params"]),
      "decoder": DiffPool(x_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          model_config["decoder"]["embed_rgcn_layer_params"],
                          model_config["decoder"]["pool_rgcn_layer_params"],
                          model_config["decoder"]["embed_block_layer_params"],
                          model_config["decoder"]["pool_block_layer_params"],
                          model_config["decoder"]["ff_layer_params"])
      })

  def forward(self, x, adj, module_type="encoder"):
    for name, param in self.dipol_modules[module_type].named_parameters():
      print(name, param.size())
    self.dipol_modules[module_type](x, adj)
