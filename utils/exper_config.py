from torch.cuda import is_available, manual_seed as torch_set_seed
from torch.backends.cudnn import benchmark, deterministic
from numpy.random import seed as numpy_set_seed
from os.path import join, abspath, curdir, exists
from os import mkdir
from pathlib import Path
import yaml
from tensorboardX import SummaryWriter
from data.sparse_molecular_dataset import SparseMolecularDataset
import csv


class Exper_Config:

    def __init__(self, run_type, resume, resume_step,
                 num_epochs, batch_size,
                 learning_rate, rl_lambda,
                 optimize_for, n_samples,
                 n_critic, z_dim, log_every,
                 val_chkpt_every, dataset,
                 use_cuda, model_config_file):
        self.resume = resume
        self.resume_step = resume_step
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rl_lambda = rl_lambda
        self.optimize_for = optimize_for
        self.n_samples = n_samples
        self.z_dim = z_dim
        self.n_critic = n_critic
        self.num_classes = 1
        self.log_every = log_every
        self.val_chkpt_every = val_chkpt_every
        self.dataset = dataset
        self.use_cuda = use_cuda
        self.num_vertices = 9
        self.replica_num = 0
        self._set_device()
        self.model_config_filename = model_config_file.split(".")[0]
        self._set_paths()
        self._load_dataset()
        self.summary_writer = self._get_SummaryWriter()
        self.model_configs = self._get_exper_config(model_config_file)
        self.seeds = self.model_configs["seeds"]
        self.total_replica_num = len(self.seeds)
        self._set_seeds()

    def _set_seeds(self):
        # set torch and numpy seeds
        torch_set_seed(self.seeds["torch_seeds"][self.replica_num])
        numpy_set_seed(self.seeds["numpy_seeds"][self.replica_num])

    def _set_device(self):
        if is_available() and self.use_cuda:
            self.device = "cuda"
            benchmark = False
            deterministic = True
        else:
            self.device = "cpu"

    def _set_paths(self):
        ROOT_DIR = abspath(curdir)
        DATA_DIR = join(ROOT_DIR, "data")
        RESULTS_DIR = join(ROOT_DIR, "results")
        EXPER_RESULTS_DIR = join(RESULTS_DIR, self.model_config_filename)
        MODELS_DIR = join(ROOT_DIR, "models")
        MODELS_CONFIG_DIR = join(MODELS_DIR, "configs")
        self.paths = {
            "ROOT_DIR": ROOT_DIR,
            "DATA_DIR": DATA_DIR,
            "RESULTS_DIR": RESULTS_DIR,
            "MODELS_DIR": MODELS_DIR,
            "MODELS_CONFIG_DIR": MODELS_CONFIG_DIR,
            "EXPER_RESULTS_DIR": EXPER_RESULTS_DIR,
            "EXPER_LOG_DIR": join(EXPER_RESULTS_DIR, "logs"),
            "EXPER_SUMMAIRES_DIR": join(EXPER_RESULTS_DIR, "summaries"),
            "EXPER_CHKPTS_DIR": join(EXPER_RESULTS_DIR, "chkpts"),
            "EXPER_VALMOLs_DIR": join(EXPER_RESULTS_DIR, "valmols")
        }
        for path in sorted(self.paths.values(), key=lambda x: len(x.split("/"))):
            self._mk_prjPath(path)

    def _mk_prjPath(self, path):
        if not exists(path): mkdir(path)

    def _mk_prjPaths(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self):
        if self.dataset == "qm9":
            self.data = SparseMolecularDataset()
            self.data.load(join(self.paths["DATA_DIR"], "gdb9_9nodes.sparsedataset"))

    def increment_replica_num(self):
        self.replica_num += 1
        self._set_seeds()

    def set_model_config(self, model_k):
        self.model_config = self.model_configs["expers"][model_k]

    def set_curr_exper_name(self, curr_exper_name):
        self.curr_exper_name = "{}_{}".format(self.model_config_filename,
                                              curr_exper_name)
        self.curr_exper_name_replica = "{}/{}".format(self.replica_num,
                                                      self.curr_exper_name)
        self.results_curr_exper_name_replica = self._get_resultsFile("results", "csv",
                                                                     ['step', 'run_type','NP score', 'QED score', 'logP score', 'SA score',
                                                                      'diversity score', 'drugcandidate score',
                                                                      'valid score', 'unique score', 'novel score'])
        self.time_curr_exper_name_replica = self._get_resultsFile("times", "csv",["step", "run_type", "time"])
        self.model_params_curr_exper_name_replica = self._get_resultsFile("model_params", "txt")
        self.results_curr_exper_name_replica.writeheader()
        self.time_curr_exper_name_replica.writeheader()

    def _get_exper_config(self, model_config_file):
        # assert yml extension
        config_file_path = join(self.paths["MODELS_CONFIG_DIR"],
                                model_config_file)
        assert (exists(config_file_path)), "{} does not exists".format(config_file_path)
        return yaml.load(open(join(config_file_path), "rb"))

    def _get_SummaryWriter(self):
        return SummaryWriter(self.paths["EXPER_SUMMAIRES_DIR"])

    def _get_resultsFile(self, file, out_type, fieldnames=None):
        self._mk_prjPaths(join(self.paths["EXPER_LOG_DIR"], self.curr_exper_name_replica))
        if out_type=="txt":
            return open(join(self.paths["EXPER_LOG_DIR"], self.curr_exper_name_replica, file + ".txt"), "w+")
        if out_type=="csv":
            return csv.DictWriter(open(join(self.paths["EXPER_LOG_DIR"], self.curr_exper_name_replica, file + ".txt"), "w+"), fieldnames)
    def set_chkpt_path(self, path):
        self._mk_prjPaths(path)
        return path
