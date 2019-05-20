import torch
improt torch.nn as nn
from torch import optim
from torch.autograd import Variable
from utils.utils import *

class Model_Ops:
  def __init__(self, exper_config):
    self.exper_config = exper_config
    self._model_builder()


  def _model_builder(self):
    if self.exper_config.model_config["type"] == "DiPol_GAN":
      from models.models import DiPol_GAN
      self.gen = DiPol_Gen(self.exper_config.data.atom_num_types, # TODO will change
                             self.exper_config.data.bond_num_types-1, # TODO will change
                             self.exper_config.num_vertices,
                             self.exper_config.z_dim,
                             self.exper_config.num_classes,
                             self.exper_config.model_config)
      self.dscr = DiPol_Dscr(self.exper_config.data.atom_num_types, # TODO will change
                             self.exper_config.data.bond_num_types-1, # TODO will change
                             self.exper_config.num_vertices,
                             self.exper_config.z_dim,
                             self.exper_config.num_classes,
                             self.exper_config.model_config)
      if self.exper_config.model_config["gen"]["optimizer"]=="adam":
        gen_optmz = optim.Adam(self.gen.parameters(),
                               lr=self.exper_config.learning_rate,
                               betas=(0.5, 0.999))
      if self.exper_config.model_config["dscr"]["optimizer"]=="adam":
        dscr_optmz = optim.Adam(self.dscr.parameters(),
                                lr=self.exper_config.learning_rate,
                                betas=(0.5, 0.999))
      self.dscr_loss = nn.BCELoss()
      self.gen.to(self.exper_config.device)
      self.dscr.to(self.exper_config.device)
      self.dscr_loss.to(self.exper_config.device)


  def process_batch(self, z, flogPs, rlogPs=None, a=None, x=None):
    def label2onehot(self, labels, dim, device):
      """Convert label indices to one-hot vectors."""
      out = torch.zeros(list(labels.size())+[dim]).to(self.exper_config.device)
      out.scatter_(len(out.size())-1, labels.unsqueeze(-1), 1.)
      return out

    z = torch.from_numpy(z).to(self.exper_config.device).float() # latent space
    # TODO might have to change this for other tasks
    flogPs = torch.from_numpy(flogPs).to(self.exper_config.device).float() # [b]
    output = [z, flogPs]
    if rlogPs and a and x: 
      rlogPs = torch.from_numpy(rlogPs).to(self.exper_config.device).float() # [b]
      a = torch.from_numpy(a).to(self.exper_config.device).long() # Adjacency [b, n, n]
      x = torch.from_numpy(x).to(self.exper_config.device).long() # Nodes (represented as atomic number) [b, n]
      x_tensor = self.label2onehot(x, self.exper_config.data.atom_num_types, self.exper_config.device) # [b, n, atom_types]
      a_tensor = self.label2onehot(a, self.exper_config.data.bond_num_types, self.exper_config.device) # [b, n, n, bond_types]
      output.extend([rlogPs, a, a_tensor, x_tensor])
    return output


  def unrel(self, rel_adj):
    return rel_adj.argmax(-1)


  #def log_performance(self):
    """ log performance here """
    # self.exper_config.summary_writer()


  def validate(self):
    self.model.eval()
    print("validate function")
#   mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_validation_batch(self.exper_config.data.validation_count)
#   a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)


  def test(self):
    self.model.eval()
    print("test function")
#   mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_test_batch(self.exper_config.data.validation_count)
#   a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)


  def train_dscrim(self, x, adj, rel_adj):
    return self.dscr((x,
                      adj.float(),
                      rel_adj[:,:,:,1:].permute(0,3,1,2)),
                      "dscr")


  def train(self):
    self.dscr.train()
    self.gen.eval()

    rlabel = 1
    flabel = 0
    rlabel_var = Variable(torch.FloatTensor(self.exper_config.batch_size).to(self.exper_config.device)
    flabel_var = Variable(torch.FloatTensor(self.exper_config.batch_size).to(self.exper_config.device)
    rlabel_var.fill_(rlabel)
    flabel_var.fill_(flabel)


    batches_per_epoch = self.exper_config.data.train_count//self.exper_config.batch_size
    total_training_steps = self.exper_config.num_epochs*batches_per_epoch 
    # training loop for given experiment
    for step in range(total_training_steps):
      mols, _, _, a, x, _, _, _, _, real_logPs, z, fake_logPs = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
      z, flogPs, rlogPs, adj, rel_adj, x = self.process_batch(z, flogPs, rlogPs, a, x)
      out, link_loss = self.train_dscrim(x, adj, rel_adj)

      if step % self.exper_config.train_gen == 0:
        self.dscr.eval()
        self.gen.train()
        z, flogPs = self.exper_config.data.next_train_batch_fake(self.exper_config.batch_size,
                                                                 self.exper_config.z_dim)
        z, flogPs = self.process_batch(z, flogPs)
        x, rel_adj = self.gen((z), "gen")
        adj = self.unrel(rel_adj)
        out, link_loss = self.train_dscrim(x, adj, rel_adj)

        self.dscr.train()
        self.gen.eval()

      if step % self.exper_config.validate_every == 0: 
        self.validate()
       if step % self.exper_config.log_every == 0: 
         print("log step")

    self.test()
