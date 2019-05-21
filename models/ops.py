import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from utils.utils import *
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

class Model_Ops:
  def __init__(self, exper_config):

    self.first_log=True
    self.exper_config = exper_config
    self._model_builder()
    self.chem_metrics_ = self.performance_metrics("chem_metrics")
    self.average_precision_scores_ = self.performance_metrics("average_precision_score")
    self.roc_auc_scores_ = self.performance_metrics("roc_auc_score")
    self.accuracy_scores_ = self.performance_metrics("accuracy_score")


  def _model_builder(self):
    if self.exper_config.model_config["type"] == "DiPol_GAN":
      from models.models import DiPol_Gen, DiPol_Dscr
      self.generator = DiPol_Gen(self.exper_config.data.atom_num_types, # TODO will change
                             self.exper_config.data.bond_num_types-1, # TODO will change
                             self.exper_config.num_vertices,
                             self.exper_config.z_dim,
                             self.exper_config.num_classes,
                             self.exper_config.model_config)
      self.discriminator = DiPol_Dscr(self.exper_config.data.atom_num_types, # TODO will change
                             self.exper_config.data.bond_num_types-1, # TODO will change
                             self.exper_config.num_vertices,
                             self.exper_config.z_dim,
                             self.exper_config.num_classes,
                             self.exper_config.model_config)
      if self.exper_config.model_config["gen"]["optimizer"]=="adam":
        self.generator_optmz = optim.Adam(self.generator.parameters(),
                               lr=self.exper_config.learning_rate,
                               betas=(0.5, 0.999))
      if self.exper_config.model_config["dscr"]["optimizer"]=="adam":
        self.discriminator_optmz = optim.Adam(self.discriminator.parameters(),
                                lr=self.exper_config.learning_rate,
                                betas=(0.5, 0.999))
      self.discriminator_loss = nn.BCELoss()
      self.generator.to(self.exper_config.device)
      self.discriminator.to(self.exper_config.device)
      self.discriminator_loss.to(self.exper_config.device)
      self.print_network(self.discriminator, "discriminator")
      self.print_network(self.generator, "generator")


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

  # todo taken from molgan pytorch implementation
  def print_network(self, model, name):
      """Print out the network information."""
      num_params = 0
      for p in model.parameters():
          num_params += p.numel()
      print(model)
      print(name)
      print("The number of parameters: {}".format(num_params))

  # todo taken from molgan pytorch implementation
  def restore_model(self, resume_iters):
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(resume_iters))
    G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
    D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
    V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(resume_iters))
    self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

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
      return average_precision_scores

    def accuracy_scores(labels, preds):
      return accuracy_score(labels.to("cpu").detach().numpy(),
                            preds.to("cpu").detach().numpy())
    if metric_type=="accuracy_score":
      return accuracy_scores


  def validate(self):
    # validate mode
    self.discriminator.eval()
    self.generator.eval()

    mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_validation_batch(self.exper_config.data.validation_count)
    z, a, a_tensor, x_tensor, z = self.process_batch(z, a, x)


  def test(self):
    # test mode
    self.discriminator.eval()
    self.generator.eval()

    mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_test_batch(self.exper_config.data.validation_count)
    z, a, a_tensor, x_tensor, z = self.process_batch(z, a, x)


  def train(self):



    batches_per_epoch = self.exper_config.data.train_count//self.exper_config.batch_size
    total_training_steps = self.exper_config.num_epochs*batches_per_epoch 
    # training loop for given experiment
    for step in range(total_training_steps):
      rlabel_var = Variable(torch.FloatTensor(self.exper_config.batch_size)\
                            .to(self.exper_config.device)).fill_(1) # real labels
      flabel_var = Variable(torch.FloatTensor(self.exper_config.batch_size)\
                            .to(self.exper_config.device)).fill_(0) # fake labels
      # -------------
      # train generator 
      # -------------
      if step % self.exper_config.n_critic == 0 and step is not 0:
        # gen training mode
        self.discriminator.eval()
        self.generator.train()

        # fetch and process data
        z = self.exper_config.data.get_z(self.exper_config.batch_size,
                                         self.exper_config.z_dim)
        z = self.process_batch(z)[0]
        x, rel_adj, hx, hrel_adj = self.generator((z), catsamp="hard_gumbel")
        gfdscr_preds, gfdscr_logits, gflpls, gfles = self.discriminator((x,
                                                                         rel_adj.argmax(-1).float(),
                                                                         rel_adj[:,:,:,1:]\
                                                                         .permute(0,3,1,2)))
        valid, unique, novel, diverse = self.chem_metrics_(hx, hrel_adj)
        gfdscr_loss = self.discriminator_loss(gfdscr_preds, flabel_var)
        gfdscr_loss = gfdscr_loss + gflpls + gfles
        self.generator_optmz.zero_grad()
        gfdscr_loss.backward()
        self.generator_optmz.step()
      # -------------
      # train discriminator 
      # -------------
      else:
        # dscr training mode
        self.discriminator.train()
        self.generator.eval()

        # fetch and process data
        mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
                                                                                                  self.exper_config.z_dim)
        z, adj, rel_adj, x = self.process_batch(z, a, x)
  
        # dscr with real
        rdscr_preds, rpred_logits, rlpls, rles = self.discriminator((x,
                                                                    adj.float(),
                                                                    rel_adj[:,:,:,1:]\
                                                                    .permute(0,3,1,2)))
        #rdscr_loss = self.discriminator_loss(rdscr_preds, rlabel_var)
        all_rdscr_loss = rdscr_loss + rlpls + rles
  
        # dscr with fake
        x, rel_adj = self.generator((z))
        fdscr_preds, fpred_logits, flpls, fles = self.discriminator((x,
                                                                    rel_adj.argmax(-1).float(),
                                                                    rel_adj[:,:,:,1:]\
                                                                    .permute(0,3,1,2)))
        fdscr_loss = self.discriminator_loss(fdscr_preds, flabel_var)
        all_fdscr_loss =  fdscr_loss + flpls + fles
  
        # train dscr real and fake losses
        all_dscr_losses = all_rdscr_loss + all_fdscr_loss
        self.discriminator_optmz.zero_grad()
        all_dscr_losses.backward()
        self.discriminator_optmz.step()

      if step % self.exper_config.log_every == 0 and step is not 0: 

#       gen_roc_auc_score = self.roc_auc_scores_(flabel_var, gfdscr_preds)
#       rdscr_roc_auc_score = self.roc_auc_score(rlabel_var, rdscr_preds)
#       fdscr_roc_auc_score = self.roc_auc_score(flabel_var, fdscr_preds)
#       gen_avg_pre_score = self.average_precision_score_(flabel_var, gfdscr_preds)
#       rdscr_avg_pre_score = self.average_precision_score_(rlabel_var, rdscr_preds)
#       fdscr_avg_pre_score = average_precision_score(flabel_var, fdscr_preds)
        fdscr_accuracy = self.accuracy_scores(flabel_var, fdscr_preds)
        rdscr_accuracy = self.accuracy_scores(rlabel_var, rdscr_preds)
        gen_accuracy = self.accuracy_scores(flabel_var, gfdscr_preds)

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
        self.exper_config.summary_writer.add_scalars("{}/train/accuracies".format(self.exper_config.curr_exper_name_replica),
                                                      {"fdscr_accuracy": fdscr_accuracy,
                                                       "rdscr_accuracy": rdscr_accuracy,
                                                       "gen_accuracy": gen_accuracy},
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
        for name, param in self.discriminator.named_parameters():
          if param.requires_grad == True:
            self.exper_config.summary_writer.add_histogram("{}/train/{}".format(self.exper_config.curr_exper_name_replica, 
                                                                                  name),
                                                             param,
                                                             step)
      # Save model checkpoints.
      if (i+1) % self.model_save_step == 0:
          G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
          D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
          V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))
          torch.save(self.G.state_dict(), G_path)
          torch.save(self.D.state_dict(), D_path)
          torch.save(self.V.state_dict(), V_path)
          print('Saved model checkpoints into {}...'.format(self.model_save_dir))

      if step % self.exper_config.validate_every == 0: 
        print('val')
        #self.validate()

    #self.test()
