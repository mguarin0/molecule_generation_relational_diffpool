import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from utils.utils import *
from sklearn.metrics import average_precision_score, roc_auc_score

class Model_Ops:
  def __init__(self, exper_config):
    self.exper_config = exper_config
    self._model_builder()


  def _model_builder(self):
    if self.exper_config.model_config["type"] == "DiPol_GAN":
      from models.models import DiPol_Gen, DiPol_Dscr
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
        self.gen_optmz = optim.Adam(self.gen.parameters(),
                               lr=self.exper_config.learning_rate,
                               betas=(0.5, 0.999))
      if self.exper_config.model_config["dscr"]["optimizer"]=="adam":
        self.dscr_optmz = optim.Adam(self.dscr.parameters(),
                                lr=self.exper_config.learning_rate,
                                betas=(0.5, 0.999))
      self.dscr_loss = nn.BCELoss()
      self.gen.to(self.exper_config.device)
      self.dscr.to(self.exper_config.device)
      self.dscr_loss.to(self.exper_config.device)


  def process_batch(self, z, flogPs, rlogPs=None, a=None, x=None):
    def label2onehot(labels, dim, device):
      """Convert label indices to one-hot vectors."""
      out = torch.zeros(list(labels.size())+[dim]).to(self.exper_config.device)
      out.scatter_(len(out.size())-1, labels.unsqueeze(-1), 1.)
      return out

    z = torch.from_numpy(z).to(self.exper_config.device).float() # latent space
    # TODO might have to change this for other tasks
    flogPs = torch.from_numpy(flogPs).to(self.exper_config.device).float() # [b]
    output = [z, flogPs]
    if rlogPs is not None and a is not None and x is not None: 
      rlogPs = torch.from_numpy(rlogPs).to(self.exper_config.device).float() # [b]
      a = torch.from_numpy(a).to(self.exper_config.device).long() # Adjacency [b, n, n]
      x = torch.from_numpy(x).to(self.exper_config.device).long() # Nodes (represented as atomic number) [b, n]
      x_tensor = label2onehot(x, self.exper_config.data.atom_num_types, self.exper_config.device) # [b, n, atom_types]
      a_tensor = label2onehot(a, self.exper_config.data.bond_num_types, self.exper_config.device) # [b, n, n, bond_types]
      output.extend([rlogPs, a, a_tensor, x_tensor])
    return output


  def unrel(self, rel_adj):
    return rel_adj.argmax(-1)


  #def log_performance(self):
    """ log performance here """
    # self.exper_config.summary_writer()


  def validate(self):
    print("validate function")
#   mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_validation_batch(self.exper_config.data.validation_count)
#   a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)


  def test(self):
    print("test function")
#   mols, _, _, a, x, _, _, _, _, z, _ = self.exper_config.data.next_test_batch(self.exper_config.data.validation_count)
#   a_tensor, x_tensor, z, real_logPs = self.process_batch(mols, a, x, z)


  def run_dscr(self, x, adj, rel_adj):
    return self.dscr((x,
                      adj.float(),
                      rel_adj[:,:,:,1:].permute(0,3,1,2)))


  def train(self):
    # dscr training mode
    self.dscr.train()
    self.gen.eval()

    rlabel = 1
    flabel = 0
    rlabel_var = Variable(torch.FloatTensor(self.exper_config.batch_size).to(self.exper_config.device))
    flabel_var = Variable(torch.FloatTensor(self.exper_config.batch_size).to(self.exper_config.device))
    rlabel_var.fill_(rlabel)
    flabel_var.fill_(flabel)


    batches_per_epoch = self.exper_config.data.train_count//self.exper_config.batch_size
    total_training_steps = self.exper_config.num_epochs*batches_per_epoch 
    # training loop for given experiment
    for step in range(total_training_steps):
      # fetch and process data
      mols, _, _, a, x, _, _, _, _, rlogPs, flogPs, z = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
                                                                                                self.exper_config.z_dim)
      z, flogPs, rlogPs, adj, rel_adj, x = self.process_batch(z, flogPs, rlogPs, a, x)

      # dscr with real
      rdscr_preds, rpred_logits, rlpls, rles = self.run_dscr(x, adj, rel_adj)
      rdscr_loss = self.dscr_loss(rdscr_preds, rlabel_var)
      all_rdscr_loss = rdscr_loss + rlpls + rles

      # dscr with fake
      x, rel_adj = self.gen((z))
      adj = self.unrel(rel_adj)
      fdscr_preds, fpred_logits, flpls, fles = self.run_dscr(x, adj, rel_adj)
      fdscr_loss = self.dscr_loss(fdscr_preds, flabel_var)
      all_fdscr_loss =  fdscr_loss + flpls + fles

      # train dscr real and fake losses
      all_dscr_losses = all_rdscr_loss + all_fdscr_loss
      self.dscr_optmz.zero_grad()
      all_dscr_losses.backward()
      self.dscr_optmz.step()

      if step % self.exper_config.train_gen == 0:
        # gen training mode
        self.dscr.eval()
        self.gen.train()

        # fetch and process data
        flogPs, z = self.exper_config.data.next_train_batch_fake(self.exper_config.batch_size,
                                                                 self.exper_config.z_dim)

        z, flogPs = self.process_batch(z, flogPs)
        x, rel_adj = self.gen((z))
        adj = self.unrel(rel_adj)
        gfdscr_preds, gfdscr_logits, gflpls, gfles = self.run_dscr(x, adj, rel_adj)
        gfdscr_loss = self.dscr_loss(gfdscr_preds, flabel_var)
        gfdscr_loss = gfdscr_loss + gflpls + gfles
        self.gen_optmz.zero_grad()
        gfdscr_loss.backward()
        self.gen_optmz.step()

        # switch back to dscr training mode
        self.dscr.train()
        self.gen.eval()

      if step % self.exper_config.log_every == 0: 
         gen_roc_auc_score = roc_auc_score(flabel_var.to("cpu").detach().numpy(), gfdscr_preds.to("cpu").detach().numpy())
         gen_avg_pre_score = average_precision_score(flabel_var.to("cpu").detach().numpy(), gfdscr_preds.to("cpu").detach().numpy())
         rdscr_roc_auc_score = roc_auc_score(rlabel_var.to("cpu").detach().numpy(), rdscr_preds.to("cpu").detach().numpy())
         rdscr_avg_pre_score = average_precision_score(rlabel_var.to("cpu").detach().numpy(), rdscr_preds.to("cpu").detach().numpy())
         fdscr_roc_auc_score = roc_auc_score(flabel_var.to("cpu").detach().numpy(), fdscr_preds.to("cpu").detach().numpy())
         fdscr_avg_pre_score = average_precision_score(flabel_var.to("cpu").detach().numpy(), fdscr_preds.to("cpu").detach().numpy())


         print("=========== TRAIN NEW RESULTS ===========")
         print("|| REAL DSCR || step: {} || rdscr_loss: {} || rlpls: {} || rles: {}".format(step, rdscr_loss,  rlpls, rles))
         print("|| FAKE DSCR || step: {} || fdscr_loss: {} || flpls: {} || fles: {}".format(step, fdscr_loss,  flpls, fles))
         print("|| FAKE GEN  || step: {} || gfdscr_loss: {} || gflpls: {} || gfles: {}".format(step, gfdscr_loss,  gflpls, gfles))
         self.exper_config.summary_writer.add_scalars("{}/train/avg_pre_score".format(self.exper_config.curr_exper_name_replica), 
                                                      {"gen_avg_pre_score": gen_avg_pre_score.,
                                                       "rdscr_avg_pre_score": rdscr_avg_pre_score,
                                                       "fdscr_avg_pre_score": fdscr_avg_pre_score},
                                                      step)
         self.exper_config.summary_writer.add_scalars("{}/train/roc_auc_score".format(self.exper_config.curr_exper_name_replica), 
                                                      {"gen_roc_auc_score": gen_roc_auc_score,
                                                       "rdscr_roc_auc_score": rdscr_roc_auc_score,
                                                       "fdscr_roc_auc_score": fdscr_roc_auc_score},
                                                      step)
         self.exper_config.summary_writer.add_scalars("{}/train/gen_losses".format(self.exper_config.curr_exper_name_replica), 
                                                      {"gfdscr_loss": gfdscr_loss.to("cpu").detach().numpy(),
                                                       "gflpl": gflpls.to("cpu").detach().numpy(),
                                                       "gfle": gfles.to("cpu").detach().numpy()},
                                                      step)
         self.exper_config.summary_writer.add_scalars("{}/train/real_dscr_losses".format(self.exper_config.curr_exper_name_replica), 
                                                      {"rdscr_loss": rdscr_loss.to("cpu").detach().numpy(),
                                                       "rlpl": rlpls.to("cpu").detach().numpy(),
                                                       "rle": rles.to("cpu").detach().numpy()},
                                                      step)
         self.exper_config.summary_writer.add_scalars("{}/train/fake_dscr_losses".format(self.exper_config.curr_exper_name_replica), 
                                                      {"fdscr_loss":fdscr_loss.to("cpu").detach().numpy(),
                                                       "flpl": flpls.to("cpu").detach().numpy(),
                                                       "fle": fles.to("cpu").detach().numpy()},
                                                      step)

          for name, param in self.dscr.named_parameters():
            if param.requires_grad == True:
              self.exper_config.summary_writer.add_histogram("{}/train/{}".format(self.exper_config.curr_exper_name_replica, 
                                                                                  name),
                                                             param,
                                                             step)

      if step % self.exper_config.validate_every == 0: 
        self.validate()

    self.test()
