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

    def forward(self, input, catsamp=None, dem=1.0):
        x_logits, rel_adj_logits = self.gen(input)

        rel_adj = F.softmax(rel_adj_logits / dem, -1)
        x = F.softmax(x_logits / dem, -1)

        # TODO make this more consice
        def gum(x_logits, rel_adj_logits):
            gum_rel_adj = F.gumbel_softmax(rel_adj_logits.contiguous() \
                                           .view(-1, rel_adj_logits.size(-1)) / dem, hard=False) \
                .view(rel_adj_logits.size())
            gum_x = F.gumbel_softmax(x_logits.contiguous() \
                                     .view(-1, x_logits.size(-1)) / dem, hard=False) \
                .view(x_logits.size())
            return gum_x, gum_rel_adj

        def hard_gum(x_logits, rel_adj_logits):
            h_gum_rel_adj = F.gumbel_softmax(rel_adj_logits.contiguous() \
                                             .view(-1, rel_adj_logits.size(-1)) / dem, hard=True) \
                .view(rel_adj_logits.size())
            h_gum_x = F.gumbel_softmax(x_logits.contiguous() \
                                       .view(-1, x_logits.size(-1)) / dem, hard=True) \
                .view(x_logits.size())
            return h_gum_x, h_gum_rel_adj

        if catsamp == "gumbel":
            hx, hrel_adj = gum(x_logits, rel_adj_logits)
            return x, rel_adj, hx, hrel_adj
        elif catsamp == "hard_gumbel":
            hx, hrel_adj = hard_gum(x_logits, rel_adj_logits)
            return x, rel_adj, hx, hrel_adj
        else:
            return x, rel_adj


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
                             "dscr")

    def forward(self, input):
        value_logits, dscr_logits, lpls, les = self.dscr(input)
        return value_logits, \
               dscr_logits, \
               [torch.sum(torch.stack(lpls)), \
                torch.sum(torch.stack(les))]
