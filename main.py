import argparse

from utils.exper_config import Exper_Config
from models.ops import *

parser = argparse.ArgumentParser()
parser.add_argument("model_config_file",
                    type=str,
                    help="yaml file for model config")
parser.add_argument("--run_type",
                    default="train",
                    type=str)
parser.add_argument("--resume",
                    default=False,
                    type=bool)
parser.add_argument("--resume_step",
                    default=0,
                    type=int)
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
                    default="validity,dc",
                    type=str)
parser.add_argument("--n_samples",
                    default=6400,
                    type=int)
parser.add_argument("--n_critic",
                    default=5,
                    type=int)
parser.add_argument("--z_dim",
                    default=32,
                    type=int)
parser.add_argument("--log_every",
                    default=256,
                    type=int)
parser.add_argument("--val_chkpt_every",
                    default=2048,
                    help="this value should be greater than validate_every",
                    type=int)
parser.add_argument("--dataset",
                    default="qm9",
                    type=str)
parser.add_argument("--use_cuda",
                    default=True,
                    type=bool)
args = parser.parse_args()

if __name__ == "__main__":

  exper_config = Exper_Config(**vars(args))

  if args.run_type == "train":
    # run all experiments
    for model_k in exper_config.model_configs["expers"]:
      exper_config.set_curr_exper_name(model_k)
      exper_config.set_model_config(model_k)

      # run all replicas for a given experiment
      for curr_replica_num in enumerate(range(exper_config.total_replica_num)):

        # set up model operations for new replica
        model_ops = Model_Ops(exper_config)
        model_ops.train(args.resume, args.resume_step)
        exper_config.increment_replica_num()
