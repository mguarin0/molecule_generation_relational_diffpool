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
    self.chem_metrics_ = self.performance_metrics("chem_metrics")
    self.average_precision_scores_ = self.performance_metrics("average_precision_score")
    self.roc_auc_scores_ = self.performance_metrics("roc_auc_score")


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


  def process_batch(self, z, a=None, x=None):
    def label2onehot(labels, dim, device):
      """Convert label indices to one-hot vectors."""
      out = torch.zeros(list(labels.size())+[dim]).to(self.exper_config.device)
      out.scatter_(len(out.size())-1, labels.unsqueeze(-1), 1.)
      return out

    z = torch.from_numpy(z).to(self.exper_config.device).float() # latent space
    # TODO might have to change this for other tasks
    output = [z]
    if a is not None and x is not None: 
      a = torch.from_numpy(a).to(self.exper_config.device).long() # Adjacency [b, n, n]
      x = torch.from_numpy(x).to(self.exper_config.device).long() # Nodes (represented as atomic number) [b, n]
      x_tensor = label2onehot(x, self.exper_config.data.atom_num_types, self.exper_config.device) # [b, n, atom_types]
      a_tensor = label2onehot(a, self.exper_config.data.bond_num_types, self.exper_config.device) # [b, n, n, bond_types]
      output.extend([a, a_tensor, x_tensor])
    return output


  def run_dscr(self, x, adj, rel_adj):
    return self.dscr((x,
                      adj.float(),
                      rel_adj[:,:,:,1:].permute(0,3,1,2)))


  def performance_metrics(self, metric_type):
    def chem_metrics(hx, hrel_adj):
      """ pass in hard gumbel softmaxes for x and rel_adj"""
      mols = [self.exper_config.data.matrices2mol(node.data.to("cpu").numpy(), edge.data.to("cpu").numpy(), strict=True)
                for node, edge in zip(hx.argmax(-1), hrel_adj.argmax(-1))]
      valid = MolecularMetrics.valid_scores(mols)
      unique = MolecularMetrics.unique_scores(mols)
      novel = MolecularMetrics.novel_scores(mols, self.exper_config.data)
      diverse = MolecularMetrics.diversity_scores(mols, self.exper_config.data)
      return valid, unique, novel, diverse
    if metric_type=="chem_metrics":
      return chem_metrics

    def roc_auc_scores(labels, preds):
      return roc_auc_score(labels.to("cpu").detach().numpy(),
                           preds.to("cpu").detach().numpy())
    if metric_type=="roc_auc_score":
      return roc_auc_scores

    def average_precision_scores(labels, preds):
      return average_precision_score(labels.to("cpu").detach().numpy(),
                                     preds.to("cpu").detach().numpy())
    if metric_type=="average_precision_score":
      average_precision_scores


  def validate(self):
    # validate mode
    self.dscr.eval()
    self.gen.eval()

    mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_validation_batch(self.exper_config.data.validation_count)
    z, a, a_tensor, x_tensor, z = self.process_batch(z, a, x)


  def test(self):
    # test mode
    self.dscr.eval()
    self.gen.eval()

    mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_test_batch(self.exper_config.data.validation_count)
    z, a, a_tensor, x_tensor, z = self.process_batch(z, a, x)


  def train(self):

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

      # train gen 
      if step % self.exper_config.train_gen == 0 and step is not 0:
        # gen training mode
        self.dscr.eval()
        self.gen.train()

        # fetch and process data
        z = self.exper_config.data.get_z(self.exper_config.batch_size,
                                         self.exper_config.z_dim)
        z = self.process_batch(z)[0]
        x, rel_adj, hx, hrel_adj = self.gen((z), catsamp="hard_gumbel")
        gfdscr_preds, gfdscr_logits, gflpls, gfles = self.run_dscr(x,
                                                                   rel_adj.argmax(-1),
                                                                   rel_adj)
        valid, unique, novel, diverse = self.chem_metrics_(hx, hrel_adj)
        gfdscr_loss = self.dscr_loss(gfdscr_preds, flabel_var)
        gfdscr_loss = gfdscr_loss + gflpls + gfles
        self.gen_optmz.zero_grad()
        gfdscr_loss.backward()
        self.gen_optmz.step()
      # train dscr
      else:
        # dscr training mode
        self.dscr.train()
        self.gen.eval()

        # fetch and process data
        mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
                                                                                                  self.exper_config.z_dim)
        z, adj, rel_adj, x = self.process_batch(z, a, x)
  
        # dscr with real
        rdscr_preds, rpred_logits, rlpls, rles = self.run_dscr(x, adj, rel_adj)
        rdscr_loss = self.dscr_loss(rdscr_preds, rlabel_var)
        all_rdscr_loss = rdscr_loss + rlpls + rles
  
        # dscr with fake
        x, rel_adj = self.gen((z))
        fdscr_preds, fpred_logits, flpls, fles = self.run_dscr(x, rel_adj.argmax(-1), rel_adj)
        fdscr_loss = self.dscr_loss(fdscr_preds, flabel_var)
        all_fdscr_loss =  fdscr_loss + flpls + fles
  
        # train dscr real and fake losses
        all_dscr_losses = all_rdscr_loss + all_fdscr_loss
        self.dscr_optmz.zero_grad()
        all_dscr_losses.backward()
        self.dscr_optmz.step()

      if step % self.exper_config.log_every == 0 and step is not 0: 
#       gen_roc_auc_score = self.roc_auc_scores_(flabel_var, gfdscr_preds)
#       rdscr_roc_auc_score = self.roc_auc_score(rlabel_var, rdscr_preds)
#       fdscr_roc_auc_score = self.roc_auc_score(flabel_var, fdscr_preds)
#       gen_avg_pre_score = self.average_precision_score_(flabel_var, gfdscr_preds)
#       rdscr_avg_pre_score = self.average_precision_score_(rlabel_var, rdscr_preds)
#       fdscr_avg_pre_score = average_precision_score(flabel_var, fdscr_preds)

        print("=========== TRAIN NEW RESULTS ===========")
        print("|| REAL DSCR || step: {} || rdscr_loss: {} || rlpls: {} || rles: {}".format(step, rdscr_loss,  rlpls, rles))
        print("|| FAKE DSCR || step: {} || fdscr_loss: {} || flpls: {} || fles: {}".format(step, fdscr_loss,  flpls, fles))
        print("|| FAKE GEN  || step: {} || gfdscr_loss: {} || gflpls: {} || gfles: {}".format(step, gfdscr_loss,  gflpls, gfles))
        self.exper_config.summary_writer.add_scalars("{}/train/chem_metrics".format(self.exper_config.curr_exper_name_replica),
                                                      {"valid": np.mean(valid),
                                                       "unique": np.mean(unique),
                                                       "novel": np.mean(novel),
                                                       "diverse": np.mean(diverse)},
                                                      step)
#       self.exper_config.summary_writer.add_scalars("{}/train/avg_pre_score".format(self.exper_config.curr_exper_name_replica), 
#                                                     {"gen_avg_pre_score": gen_avg_pre_score,
#                                                      "rdscr_avg_pre_score": rdscr_avg_pre_score,
#                                                      "fdscr_avg_pre_score": fdscr_avg_pre_score},
#                                                     step)
#       self.exper_config.summary_writer.add_scalars("{}/train/roc_auc_score".format(self.exper_config.curr_exper_name_replica), 
#                                                     {"gen_roc_auc_score": gen_roc_auc_score,
#                                                      "rdscr_roc_auc_score": rdscr_roc_auc_score,
#                                                      "fdscr_roc_auc_score": fdscr_roc_auc_score},
#                                                     step)
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
        print('val')
        #self.validate()

    #self.test()
