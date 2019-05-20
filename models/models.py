import torch.nn as nn
from models.layers import *

class DiPol_GAN(nn.Module):
  def __init__(self, d_dim, r_dim, n_dim, z_dim, num_classes, model_config):
    super(DiPol_GAN, self).__init__()

    # encoder definition
    self.dipol_modules = nn.ModuleDict([
      ["discriminator", DiffPool(d_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          num_classes,
                          model_config["disc"]["embed_rgcn_layer_params"],
                          model_config["disc"]["pool_rgcn_layer_params"],
                          model_config["disc"]["embed_gcn_layer_params"],
                          model_config["disc"]["pool_gcn_layer_params"],
                          model_config["disc"]["ff_layer_params"],
                          "discriminator")],
      ["generator", DiffPool(d_dim,
                          r_dim,
                          n_dim,
                          z_dim,
                          num_classes,
                          model_config["gen"]["embed_rgcn_layer_params"],
                          model_config["gen"]["pool_rgcn_layer_params"],
                          model_config["gen"]["embed_gcn_layer_params"],
                          model_config["gen"]["pool_gcn_layer_params"],
                          model_config["gen"]["ff_layer_params"],
                          "generator")]
      ])

  def forward(self, input, module_type):
    return self.dipol_modules[module_type](input)

