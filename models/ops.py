import torch
from utils.utils import *

class Model_Ops:
  def __init__(self, exper_config):
    self.exper_config = exper_config
    self._model_builder()

  def label2onehot(self, labels, dim, device):
    """Convert label indices to one-hot vectors."""
    out = torch.zeros(list(labels.size())+[dim]).to(self.exper_config.device)
    out.scatter_(len(out.size())-1, labels.unsqueeze(-1), 1.)
    return out

  def process_batch(self, mols, a, x, z):
    a = torch.from_numpy(a).to(self.exper_config.device).long() # Adjacency [b, n, n]
    x = torch.from_numpy(x).to(self.exper_config.device).long() # Nodes (represented as atomic number) [b, n]
    z = torch.from_numpy(z).to(self.exper_config.device).float() # latent space
    x_tensor = self.label2onehot(x, self.exper_config.data.atom_num_types, self.exper_config.device) # [b, n, atom_types]
    a_tensor = self.label2onehot(a, self.exper_config.data.bond_num_types, self.exper_config.device) # [b, n, n, bond_types]

    # TODO might have to change this for other tasks
    real_logPs = MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
    real_logPs = torch.from_numpy(real_logPs).to(self.exper_config.device).float() # [b]

    return a_tensor, x_tensor, z, real_logPs 

  #def log_performance(self):
    """ log performance here """
    # self.exper_config.summary_writer()

  def train(self):
    #self.model

    batches_per_epoch = self.exper_config.data.train_count//self.exper_config.batch_size
    total_training_steps = self.exper_config.num_epochs*batches_per_epoch 
    # training loop for given experiment
    for step in range(total_training_steps):
      mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
                                                                                   self.exper_config.z_dim)
      a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)
      #x_tensor.size()) [32, 9, 5]
      #a_tensor.size()) [32, 9, 9, 5]

#     if step % self.exper_config.validate_every == 0: 
#       self.validate()
#     #if step % self.exper_config.log_every == 0: 

#   self.test()

# def validate(self):
#   self.model.eval()
#   mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_validation_batch(self.exper_config.data.validation_count)
#   a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)

# def test(self):
#   self.model.eval()
#   mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_test_batch(self.exper_config.data.validation_count)
#   a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)

  def _model_builder(self):
    if self.exper_config.model_config["type"] == "DiPol_GraphVAE":
      from models.models import DiPol_GraphVAE
      self.model = DiPol_GraphVAE(self.exper_config.data.atom_num_types, # TODO will change
                                  self.exper_config.data.bond_num_types, # TODO will change
                                  self.exper_config.z_dim,
                                  self.exper_config.model_config)
