import torch.nn as nn
from models.layers import *

class DiPol_GAN(nn.Module):
  def __init__(self, x_dim, r_dim, n_dim, z_dim, num_classes, model_config):
    super(DiPol_GAN, self).__init__()

    # encoder definition
    self.dipol_modules = nn.ModuleDict([
      ["discriminator", DiffPool(x_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          num_classes,
                          model_config["encoder"]["embed_rgcn_layer_params"],
                          model_config["encoder"]["pool_rgcn_layer_params"],
                          model_config["encoder"]["embed_gcn_layer_params"],
                          model_config["encoder"]["pool_gcn_layer_params"],
                          model_config["encoder"]["ff_layer_params"],
                          "discriminator")],
      ["generator", DiffPool(x_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          num_classes,
                          model_config["decoder"]["embed_rgcn_layer_params"],
                          model_config["decoder"]["pool_rgcn_layer_params"],
                          model_config["decoder"]["embed_gcn_layer_params"],
                          model_config["decoder"]["pool_gcn_layer_params"],
                          model_config["decoder"]["ff_layer_params"],
                          "generator")]
      ])

  def forward(self, input, module_type):
    return self.dipol_modules[module_type](input)

