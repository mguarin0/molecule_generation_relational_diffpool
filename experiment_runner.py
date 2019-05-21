# call with CUDA_VISIBLE_DEVICES=0

import os
import itertools
from subprocess import run, PIPE
from itertools import chain
from utils.utils import seeds

params = {"experiment_num": 0,
          "batch_dim": 32,
          "la": 1.0,
          "dropout": 0.0,
          "n_critic": 5,
          "n_samples": 6400,
          "z_dim": 32,
          "save_every": 5,
          "metric": "validity,sas",
          "epochs": 30,
          "learning_rate": 1e-3,
          "resultsfile": "results.txt",
          "np_seed": 0,
          "tf_seed": 0
}


def run_cli_cmd(cmd, err_file): return run(cmd, stdin=PIPE, stdout=PIPE, stderr=err_file, universal_newlines=True)

base_cmd = ["python", "example.py"]
la_scores = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0]
dropout_probs = [0.0, 0.1, 0.25]
num_replicas_per_experiment = 5

# sas => synthetizibility, logp => solubility, dc => drugcandidate_scores
#for op_property in ["sas", "logp", "dc"]:
for op_property in ["dc"]:
  params["metric"] = "validity,{}".format(op_property)
  params["resultsfile"] = "1_{}_{}".format(op_property, params["resultsfile"])
  err_file = open("1_{}_errs.txt".format(op_property), "a+")
  for la_score in la_scores:
    params["la"] = la_score
    for dropout_val in dropout_probs:
      params["dropout"] = dropout_val
      for replica_num in range(num_replicas_per_experiment):
        try:
          params["np_seed"] = seeds[replica_num]["np_seed"]
          params["tf_seed"] = seeds[replica_num]["tf_seed"]
          params["replica_num"] = replica_num
          results = run_cli_cmd(base_cmd+list(chain.from_iterable([["--{}".format(k), str(v)] for k, v in params.items()])),
                                err_file)
        except KeyboardInterrupt:
          exit(0)
        except:
          err_file.write("error thrown on experiment_num: {} replica_num: {}".format(params["experiment_num"],
                                                                                     params["replica_num"]))
          pass
      params["experiment_num"] = params["experiment_num"] + 1
