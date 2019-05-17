import torch.nn as nn
from layers import *

class DiPol_GraphVAE(nn.Module):
  def __init__(self, input_x_dim, model_config):
    super(DiPol_GraphVAE, self).__init__()

    # encoder definition
    self.encoder = DiffPool(input_x_dim,
                            9, # TODO find a way to put this on data ops
                            model_config["encoder"].embed_rgcn_layer_hidden_dims,
                            model_config["encoder"].embed_layer_hidden_dims,
                            model_config["encoder"].pool_layer_downsample_percents,
                            model_config["encoder"].ff_layer_hidden_dims)
    # encoder definition
    self.encoder = DiffPool(z_dim,
                            9, # TODO find a way to put this on data ops
                            model_config["decoder"].embed_rgcn_layer_hidden_dims,
                            model_config["decoder"].embed_layer_hidden_dims,
                            model_config["decoder"].pool_layer_downsample_percents)
                            model_config["decoder"].ff_layer_hidden_dims)

  def forward(self, x):
    mu, logvar = self.encode(x.view(-1, 784))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar
