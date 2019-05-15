from torch.cuda import is_available, manual_seed as torch_set_seed
from torch.backends.cudnn import benchmark, deterministic
from numpy.random import seed as numpy_set_seed
from os.path import join, abspath, curdir, exists
from os import mkdir
import yaml
from tensorboardX import SummaryWriter

class config:

  def __init__(self, total_replica_num, use_cuda, exper_name):
    self.total_replica_num = total_replica_num
    self.replica_num = 0
    self.use_cuda = use_cuda
    self.exper_name = exper_name
    self.seeds = {
    "torch_seed": [950148, 171376, 620516,
                   952498, 528800, 382510,
                   685351, 910227, 881256,
                   160444, 550690, 240362,
                   619656, 114047, 774876,
                   707406, 479884, 851298,
                   429968, 309134],
    "numpy_seed": [700112, 118184, 721042,
                   849212, 447234, 640394,
                   471836, 792048, 194291,
                   210490, 735049, 607747,
                   859680, 950268, 960985,
                   867656, 104550, 721576,
                   503440, 718723]
    }

    self.num_atoms: 9

    self._set_seeds()
    self._set_device()
    self._set_paths()

  def _set_seeds(self):
    # set torch and numpy seeds
    for seed_type in self.seeds.keys():
      seed_len = len(self.seeds[seed_type])
      assert(self.replica_num < seed_len), "error! seed type: {} \
                                            replica_num: {} \
                                            exceeds number of seeds {}".format(self.replica_num,
                                                                               seed_len)
    torch_set_seed(self.seeds["torch_seed"][self.replica_num])
    numpy_set_seed(self.seeds["numpy_seed"][self.replica_num])

  def _set_device(self):
    if is_available() and self.use_cuda:
      self.device = "cuda"
      benchmark = False
      deterministic = True
    else:
      self.device = "cpu"

  def _set_paths(self):
    ROOT_DIR = abspath(curdir)
    LIB_DIR = join(ROOT_DIR, "lib") 
    DATA_DIR = join(ROOT_DIR, "data")
    EXPER_CONFIG_DIR = join(ROOT_DIR, "exper") 
    EXPER_RESULTS_DIR = join(ROOT_DIR, "results") 
    CURR_EXPER_RESULTS_DIR = join(EXPER_RESULTS_DIR,
                                  self.exper_name) 
    self.paths = {
            "ROOT_DIR": ROOT_DIR, 
            "LIB_DIR": LIB_DIR, 
            "DATA_DIR": DATA_DIR, 
            "EXPER_CONFIG_DIR": EXPER_CONFIG_DIR,
            "EXPER_RESULTS_DIR": EXPER_RESULTS_DIR,
            "CURR_EXPER_RESULTS_DIR": CURR_EXPER_RESULTS_DIR,
            "CURR_EXPER_LOG_DIR": join(CURR_EXPER_RESULTS_DIR, "logs"), 
            "CURR_EXPER_SUMMAIRES_DIR": join(CURR_EXPER_RESULTS_DIR, "summaries"), 
            "CURR_EXPER_CHKPTS_DIR": join(CURR_EXPER_RESULTS_DIR, "chkpts")
            }
    for path in sorted(self.paths.values(), key=lambda x: len(x.split("/"))):
      self._mk_prjPaths(path)

  def _mk_prjPaths(self, path):
    if not exists(path): mkdir(path)

  def increment_replica_num(self):
    self.replica_num+=1
    self._set_seeds()

  def get_exper_config(self, exper_config_filename):
    # assert yml extension
    return yaml.load(open(join(self.paths["EXPER_CONFIG_DIR"],
  	                       exper_config_filename),
                          "rb"))

  def get_SummaryWriter(self):
    return SummaryWriter(self.paths["CURR_EXPER_SUMMAIRES_DIR"])
