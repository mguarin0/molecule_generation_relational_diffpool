from os.path import join, abspath, curdir, exists
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import logging
from copy import deepcopy

from data.sparse_molecular_dataset import SparseMolecularDataset
from utils import *
from models.train import *
"""
TODOs
* pull in cli args DONE
* set up filesystem DONE
* pull in yaml config file DONE
* set up filesystem specific to experiment run DONE
* send config object into experiment runner to configure model/run the experiment
"""
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs",
                    default=30,
                    type=int)
parser.add_argument("--batch_size",
                    default=32,
                    type=int)
parser.add_argument("--learning_rate",
                    default=1e-3,
                    type=float)
parser.add_argument("--la",
                    default=0.0,
                    type=float)
parser.add_argument("--metric",
                    default="validity,sas",
                    type=str)
parser.add_argument("--n_samples",
                    default=6400,
                    type=int)
parser.add_argument("--z_dim",
                    default=32,
                    type=int)
parser.add_argument("--chkpt_every",
                    default=5,
                    type=int)
parser.add_argument("--total_replica_num",
                    default=5,
                    type=int)
parser.add_argument("--dataset",
                    default="qm9",
                    type=str)
parser.add_argument("--use_cuda",
                    default=True,
                    type=bool)
parser.add_argument("--exper_config_filename",
                    type=str,
                    choices=["config_1.yml"],
                    help="yaml configuration file for current experiment")
args = parser.parse_args()

if __name__ == "__main__":

  exper_name = args.experiment_config.split(".")[0]
  config = config(args.total_replica_num, args.use_cuda, exper_name)
  model_config = config.get_exper_config(args.exper_config_filename)
  summary_writer = config.get_SummaryWriter()

  if args.dataset=="qm9":
    data = SparseMolecularDataset()
    data.load(join(config.paths["DATA_DIR"], "gdb9_9nodes.sparsedataset"))

  # run all experiments
  for model_k, model_v in model_config["exper"].items():
    model_prefix = "{}_{}".format(exper_name, model_k) 

    # run all replicas for a given experiment
    for curr_replica_num in enumerate(range(config.total_replica_num)):

      logging.basicConfig(filename=os.path.join(paths.EXPERIMENT_CONFIG_LOG_DIR,
                                                "{}_training.log".format(replica_num)),
                          filemode="w",
                          level=logging.DEBUG)

      model_ops = Model_Ops(data, args.num_epochs)
      # set up model for new replica
      Disc_diff_pool = DiffPool(data.atom_num_types,
                                experiment_config["experiments"][experiment_num]["gcn_layer_hidden_dims"],
                                experiment_config["experiments"][experiment_num]["ff_layer_hidden_dims"],
                                experiment_config["num_atoms"],
                                1, # TODO num_classes figure out how to handle this
                                experiment_config["experiments"][experiment_num]["layer_downsample_percents"])
      Disc_diff_pool.to(experiment_config["device"])
      disc_loss = nn.BCELoss().to(experiment_config["device"]) # discriminator loss
      aux_loss = nn.MSELoss().to(experiment_config["device"]) # aux loss

      # training loop for given experiment
      for epoch in range(experiment_config["num_epochs"]):
        for step in range(data.train_count//args.batch_size):
          mols, _, _, a, x, _, _, _, _, z, fake_logPs = data.next_train_batch(args.batch_size,
                                                                              args.z_dim)
          a = torch.from_numpy(a).to(experiment_config["device"]).long() # Adjacency [b, n, n]
          x = torch.from_numpy(x).to(experiment_config["device"]).long() # Nodes (represented as atomic number) [b, n]
          z = torch.from_numpy(z).to(experiment_config["device"]).float() # latent space
          #adj_tensor = label2onehot(a, data.bond_num_types, experiment_config["cuda"]) # [b, n, n, b_types]
          x_tensor = label2onehot(x, data.atom_num_types, experiment_config["device"]) # [b, n, atom_types]

          real_logPs = MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
          real_logPs = torch.from_numpy(real_logPs).to(experiment_config["device"]).float() # [b]
          fake_logPs = torch.from_numpy(fake_logPs).to(experiment_config["device"]).float()

          label_real_disc = deepcopy(disc_label.data.resize_(experiment_config["batch_size"]).fill_(real_label))
          label_fake_disc = deepcopy(disc_label.data.resize_(experiment_config["batch_size"]).fill_(fake_label))

          # train discriminator with real
          real_disc_out, real_aux_out = Disc_diff_pool(x_tensor, torch.clamp(a.float(), min=0, max=1), None)
          disc_loss_real = disc_loss(real_disc_out, label_real_disc)
          aux_loss_real = aux_loss(real_logPs, real_aux_out)
          disc_loss_real_all = disc_loss_real + aux_loss_real 
          disc_loss_real_all.backward()

          # train discriminator with fake 

          # TODO here figure out adj_tensor
          #if (step+1) % args.log_every == 0:
            # TODO add logging here
          #if (step+1) % args.checkpoint_every == 0:
            # TODO add checkpointing here
          # TODO log/print training results for current step
        mols, _, _, a, x, _, _, _, _ = data.next_validation_batch(data.validation_count)
        # TODO add logging here
        # TODO log/print validation results for validation step
