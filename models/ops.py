import torch

class Model_Ops:
  def __init__(self, exper_config):
    self.exper_config = exper_config

  def process_batch(self, mols, a, x, z):
    a = torch.from_numpy(a).to(self.exper_config.device).long() # Adjacency [b, n, n]
    x = torch.from_numpy(x).to(self.exper_config.device).long() # Nodes (represented as atomic number) [b, n]
    z = torch.from_numpy(z).to(self.exper_config.device).float() # latent space
    x_tensor = label2onehot(x, self.data.atom_num_types, self.exper_config.device) # [b, n, atom_types]
    a_tensor = label2onehot(a, self.data.bond_num_types, self.exper_config.device) # [b, n, n, bond_types]

    # TODO might have to change this for other tasks
    real_logPs = MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
    real_logPs = torch.from_numpy(real_logPs).to(self.exper_config.device).float() # [b]

    return a_tensor, x_tensor, z, real_logPs 

  #def log_performance(self):
    """ log performance here """
    # self.exper_config.summary_writer()

  def train(self):

    batches_per_epoch = self.exper_config.data.train_count//self.exper_config.batch_size
    total_training_steps = self.exper_config.num_epochs*batches_per_epoch 
    # training loop for given experiment
    for step in range(total_training_steps):
      mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
                                                                                   self.exper_config.z_dim)
      a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)

      if step % self.exper_config.validate_every == 0: 
        self.validate()
      #if step % self.exper_config.log_every == 0: 

    self.test()

  def validate(self):
    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch(self.data.validation_count)
    a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)

  def test(self):
    mols, _, _, a, x, _, _, _, _ = self.data.next_test_batch(self.data.validation_count)
    a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)

  def model_builder(self):
    if self.exper_config.model_config["type"] == "DiPol_GraphVAE":
      from models import DiPol_GraphVAE
      self.model = DiPol_GraphVAE(self.exper_config.data.atom_num_types,
                                  self.exper_config.data.bond_num_types,
                                  self.exper_config.data.z_dim,
                                  self.exper_config.model_config)
