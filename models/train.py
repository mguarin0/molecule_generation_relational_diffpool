import torch
from models import DiffPool

class Model_Ops:
  def __init__(self, data, num_epochs,
               batch_size, z_dim, device,
               validate_every, log_every, summary_writer,
               results_writer, exper_name):
    self.data = data
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.z_dim = z_dim
    self.device = device
    self.validate_every = validate_every
    self.log_every = log_every
    self.summary_writer = summary_writer
    self.results_writer = results_writer
    self.exper_name = exper_name

  def train(self):
    # training loop for given experiment
    for step in range(self.num_epochs*(self.data.train_count//self.batch_size)):
      mols, _, _, a, x, _, _, _, _, z, fake_logPs = self.data.next_train_batch(self.batch_size,
                                                                               self.z_dim)
      a = torch.from_numpy(a).to(self.device).long() # Adjacency [b, n, n]
      x = torch.from_numpy(x).to(self.device).long() # Nodes (represented as atomic number) [b, n]
      z = torch.from_numpy(z).to(self.device).float() # latent space
      x_tensor = label2onehot(x, self.data.atom_num_types, self.device) # [b, n, atom_types]
      a_tensor = label2onehot(a, self.data.bond_num_types, self.device) # [b, n, n, bond_types]

      # TODO might have to change this for other tasks
      real_logPs = MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
      real_logPs = torch.from_numpy(real_logPs).to(self.device).float() # [b]

      if step % validate_every == 0: 
        self.validate()
      if step % log_every == 0: 

    self.test()

  def validate(self):
    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch(self.data.validation_count)
    a = torch.from_numpy(a).to(self.device).long() # Adjacency [b, n, n]
    x = torch.from_numpy(x).to(self.device).long() # Nodes (represented as atomic number) [b, n]
    z = torch.from_numpy(z).to(self.device).float() # latent space
    x_tensor = label2onehot(x, self.data.atom_num_types, self.device) # [b, n, atom_types]
    a_tensor = label2onehot(a, self.data.bond_num_types, self.device) # [b, n, n, bond_types]

  def test(self):
    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch(self.data.validation_count)
    a = torch.from_numpy(a).to(self.device).long() # Adjacency [b, n, n]
    x = torch.from_numpy(x).to(self.device).long() # Nodes (represented as atomic number) [b, n]
    z = torch.from_numpy(z).to(self.device).float() # latent space
    x_tensor = label2onehot(x, self.data.atom_num_types, self.device) # [b, n, atom_types]
    a_tensor = label2onehot(a, self.data.bond_num_types, self.device) # [b, n, n, bond_types]

class Model_Assemble:
  def __init__(self):
