from os.path import join, abspath, curdir, exists
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from copy import deepcopy

from utils.exper_config import Exper_Config
from models.ops import *

"""
TODOs
* pull in cli args DONE
* set up filesystem DONE
* pull in yamlexper_config file DONE
* set up filesystem specific to experiment run DONE
* sendexper_config object into experiment runner toexper_configure model/run the experiment
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
parser.add_argument("--rl_lambda",
                    default=0.0,
                    type=float)
parser.add_argument("--optimize_for",
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
parser.add_argument("--validate_every",
                    default=512,
                    type=int)
parser.add_argument("--dataset",
                    default="qm9",
                    type=str)
parser.add_argument("--use_cuda",
                    default=True,
                    type=bool)
parser.add_argument("--model_config_file",
                    type=str,
                    default="config_1.yml",
                    choices=["config_1.yml"],
                    help="yaml file for model config")
args = parser.parse_args()

if __name__ == "__main__":

  exper_config = Exper_Config(**vars(args))


  # run all experiments
  for model_k in exper_config.model_config["exper"]:
    exper_config.set_curr_exper_name(model_k)

    # run all replicas for a given experiment
    for curr_replica_num in enumerate(range(exper_config.total_replica_num)):

      # set up model operations for new replica
      model_ops = Model_Ops(args.num_epochs)
      #disc_loss = nn.BCELoss().to(experiment_config["device"]) # discriminator loss
      #aux_loss = nn.MSELoss().to(experiment_config["device"]) # aux loss

      # training loop for given experiment

          #if (step+1) % args.log_every == 0:
            # TODO add logging here
          #if (step+1) % args.checkpoint_every == 0:
            # TODO add checkpointing here
          # TODO log/print training results for current step
        # TODO add logging here
        # TODO log/print validation results for validation step
      # TODO set up model for new replica
