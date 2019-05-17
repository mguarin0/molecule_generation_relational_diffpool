from math import ceil
import torch.nn as nn

class dp(nn.Module):
  def __init__(self, nodes_dim=9):
    super(dp, self).__init__()
    self.node_dim = nodes_dim
    pool_down_percent = [0.90, 0.75]
    print([self.lnn_layer_dim(perc) for perc in pool_down_percent])

  def lnn_layer_dim(self, pool_percent):
    self.node_dim = ceil(self.node_dim * pool_percent)
    return self.node_dim

  def forward():
  	print("forward")