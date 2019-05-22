import torch
import os
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from utils.utils import *
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import torch.nn.functional as F

class Model_Ops:
    def __init__(self, exper_config):

        self.metric = "validity"
        self.first_log = True
        self.exper_config = exper_config
        self.chem_metrics_ = self.performance_metrics("chem_metrics")
        self.average_precision_scores_ = self.performance_metrics("average_precision_score")
        self.roc_auc_scores_ = self.performance_metrics("roc_auc_score")
        self.accuracy_scores_ = self.performance_metrics("accuracy_score")
        self.generator_beta_1 = self.exper_config.model_config["gen"]["optimizer"][1]
        self.generator_beta_2 = self.exper_config.model_config["gen"]["optimizer"][2]
        self.discriminator_beta_1 = self.exper_config.model_config["dscr"]["optimizer"][1]
        self.discriminator_beta_2 = self.exper_config.model_config["dscr"]["optimizer"][2]
        self._model_builder()

    def _model_builder(self):
        if self.exper_config.model_config["type"] == "DiPol_GAN":
            from models.models import DiPol_Gen, DiPol_Dscr
            self.generator = DiPol_Gen(self.exper_config.data.atom_num_types,  # TODO will change
                                       self.exper_config.data.bond_num_types,  # TODO will change
                                       self.exper_config.num_vertices,
                                       self.exper_config.z_dim,
                                       self.exper_config.num_classes,
                                       self.exper_config.model_config)
            self.discriminator = DiPol_Dscr(self.exper_config.data.atom_num_types,  # TODO will change
                                            self.exper_config.data.bond_num_types,  # TODO will change
                                            self.exper_config.num_vertices,
                                            self.exper_config.z_dim,
                                            self.exper_config.num_classes,
                                            self.exper_config.model_config)
            self.value = DiPol_Dscr(self.exper_config.data.atom_num_types,  # TODO will change
                                    self.exper_config.data.bond_num_types,  # TODO will change
                                    self.exper_config.num_vertices,
                                    self.exper_config.z_dim,
                                    self.exper_config.num_classes,
                                    self.exper_config.model_config)
            if self.exper_config.model_config["gen"]["optimizer"][0] == "adam":
                self.generator_optimizer = optim.Adam(self.generator.parameters(),
                                                      lr=self.exper_config.learning_rate,
                                                      betas=(self.generator_beta_1, self.generator_beta_2))
            if self.exper_config.model_config["dscr"]["optimizer"][0] == "adam":
                self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                                          lr=self.exper_config.learning_rate,
                                                          betas=(self.discriminator_beta_1, self.discriminator_beta_2))
            self.generator.to(self.exper_config.device)
            self.discriminator.to(self.exper_config.device)
            self.value.to(self.exper_config.device)
            self.print_network(self.discriminator, "discriminator")
            self.print_network(self.generator, "generator")
            self.print_network(self.value, "value")

    def process_batch(self, z, a=None, x=None):
        def label2onehot(labels, dim, device):
            """Convert label indices to one-hot vectors."""
            out = torch.zeros(list(labels.size()) + [dim]).to(self.exper_config.device)
            out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
            return out

        z = torch.from_numpy(z).to(self.exper_config.device).float()  # latent space
        # TODO might have to change this for other tasks
        output = [z]
        if a is not None and x is not None:
            a = torch.from_numpy(a).to(self.exper_config.device).long()  # Adjacency [b, n, n]
            x = torch.from_numpy(x).to(self.exper_config.device).long()  # Nodes (represented as atomic number) [b, n]
            x_tensor = label2onehot(x, self.exper_config.data.atom_num_types,
                                    self.exper_config.device)  # [b, n, atom_types]
            a_tensor = label2onehot(a, self.exper_config.data.bond_num_types,
                                    self.exper_config.device)  # [b, n, n, bond_types]
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
        G_path = os.path.join(self.exper_config.paths["EXPER_CHKPTS_DIR"], self.curr_exper_name_replica,
                              '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.exper_config.paths["EXPER_CHKPTS_DIR"], self.curr_exper_name_replica,
                              '{}-D.ckpt'.format(resume_iters))
        V_path = os.path.join(self.exper_config.paths["EXPER_CHKPTS_DIR"], self.curr_exper_name_replica,
                              '{}-V.ckpt'.format(resume_iters))
        self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.exper_config.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.exper_config.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.exper_config.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.exper_config.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def performance_metrics(self, metric_type):
        def chem_metrics(hx, hrel_adj):
            """ pass in hard gumbel softmaxes for x and rel_adj"""
            mols = [self.exper_config.data.matrices2mol(node.data.to("cpu").numpy(), edge.data.to("cpu").numpy(),
                                                        strict=True)
                    for node, edge in zip(hx, hrel_adj)]
            valid = MolecularMetrics.valid_scores(mols)
            unique = MolecularMetrics.unique_scores(mols)
            novel = MolecularMetrics.novel_scores(mols, self.exper_config.data)
            diverse = MolecularMetrics.diversity_scores(mols, self.exper_config.data)
            drugcandidate_score = MolecularMetrics.drugcandidate_scores(mols, self.exper_config.data)
            return valid, unique, novel, diverse, drugcandidate_score

        if metric_type == "chem_metrics":
            return chem_metrics

        def roc_auc_scores(labels, preds):
            return roc_auc_score(labels.to("cpu").detach().numpy(),
                                 preds.to("cpu").detach().numpy())

        if metric_type == "roc_auc_score":
            return roc_auc_scores

        def average_precision_scores(labels, preds):
            return average_precision_score(labels.to("cpu").detach().numpy(),
                                           preds.to("cpu").detach().numpy())

        if metric_type == "average_precision_score":
            return average_precision_scores

        def accuracy_scores(labels, preds):
            return accuracy_score(labels.to("cpu").detach().numpy(),
                                  preds.to("cpu").detach().numpy())

        if metric_type == "accuracy_score":
            return accuracy_scores

    def validate(self):
        # validate mode
        self.discriminator.eval()
        self.generator.eval()

        mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_validation_batch(
            self.exper_config.data.validation_count)
        z, adj, rel_adj, x = self.process_batch(z, a, x)

    def test(self):
        # test mode
        self.discriminator.eval()
        self.generator.eval()

        mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_test_batch(self.exper_config.data.test_count)
        z, adj, rel_adj, x = self.process_batch(z, a, x)

    def train(self):

        # Learning rate cache for decaying.
        g_lr = self.exper_config.learning_rate
        d_lr = self.exper_config.learning_rate

        batches_per_epoch = self.exper_config.data.train_count // self.exper_config.batch_size
        total_training_steps = self.exper_config.num_epochs * batches_per_epoch
        # training loop for given experiment
        for step in range(total_training_steps):
            real_label_var = Variable(torch.FloatTensor(self.exper_config.batch_size) \
                                      .to(self.exper_config.device)).fill_(1)  # real labels
            fake_label_var = Variable(torch.FloatTensor(self.exper_config.batch_size) \
                                      .to(self.exper_config.device)).fill_(0)  # fake labels
            mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
                                                                                      self.exper_config.z_dim)
            z, adj, rel_adj, x = self.process_batch(z, a, x)

            # ----------------------------------------------------------#
            #                     train generator                       #
            # ----------------------------------------------------------#
            if step % self.exper_config.n_critic == 0 and step is not 0:
                # gen training mode
                # -----------------------------------#
                #           train with fake          #
                # -----------------------------------#

                hx, hrel_adj = self.generator((z))  # descritize input to discriminator
                generator_train_discriminator_preds, generator_train_discriminator_logits, generator_diffpool_losses = self.discriminator(
                    (hx,
                     hrel_adj.argmax(
                         -1).float(),
                     hrel_adj[:,:,:,1:] \
                     .permute(
                         0,
                         3,
                         1,
                         2)))

                fake_generator_loss = - torch.mean(generator_train_discriminator_logits)

                # -----------------------------------#
                #           real reward              #
                # -----------------------------------#
                real_reward = torch.from_numpy(self.reward(mols)).to(self.exper_config.device)

                # -----------------------------------#
                #           fake reward              #
                # -----------------------------------#
                # TODO don't need to call generator twice (if problem check here
                _, _, hx, hrel_adj = self.generator((z), catsamp="hard_gumbel")  # descritize input to discriminator
                hx_, hrel_adj_ = hx.argmax(-1), hrel_adj.argmax(-1)
                mols = [self.exper_config.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                        for e_, n_ in zip(hrel_adj_, hx_)]
                generator_valid, generator_unique, generator_novel, generator_diverse, generator_drug_candidate_score = self.chem_metrics_(
                    hx_, hrel_adj_)
                fake_reward = torch.from_numpy(self.reward(mols)).to(self.exper_config.device)

                # Value loss
                real_value_logit, _, real_value_diff_pool_losses = self.value((x, adj.float(), rel_adj[:,:,:,1:].permute(0,3,1,2)))
                fake_value_logit, _, real_value_diff_pool_losses = self.value((hx, hrel_adj.argmax(-1).float(), hrel_adj[:,:,:,1:].permute(0,3,1,2)))
                generator_loss_value = torch.mean((F.sigmoid(real_value_logit) - real_reward) ** 2 + (
                        F.sigmoid(fake_value_logit) - fake_reward) ** 2)

                # Backward and optimize.
                generator_loss = fake_generator_loss + generator_loss_value
                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()

            # ----------------------------------------------------------#
            #                     train discriminator                   #
            # ----------------------------------------------------------#
            else:
                # -----------------------------------#
                #           train with real          #
                # -----------------------------------#
                real_discriminator_preds, real_discriminator_logits, real_discriminator_diffpool_losses = self.discriminator(
                    (x,
                     adj.float(),
                     rel_adj[:, :, :, 1:] \
                     .permute(0, 3, 1,
                              2)))
                real_discriminator_loss = - torch.mean(real_discriminator_logits)

                # -----------------------------------#
                #           train with fake          #
                # -----------------------------------#
                hx, hrel_adj = self.generator((z))
                fake_discriminator_preds, fake_discriminator_logits, fake_discriminator_diffpool_losses = self.discriminator(
                    (hx,
                     hrel_adj.argmax(-1).float(),
                     hrel_adj[:, :, :, 1:] \
                     .permute(0, 3, 1, 2)))
                fake_discriminator_loss = torch.mean(fake_discriminator_logits)

                # Compute loss for gradient penalty.
                eps = torch.rand(real_discriminator_logits.size(0), 1, 1, 1).to(self.exper_config.device)
                x_int0 = (eps * rel_adj + (1. - eps) * hrel_adj).requires_grad_(True)
                x_int1 = (eps.squeeze(-1) * x + (1. - eps.squeeze(-1)) * hx).requires_grad_(True)
                grad0, grad1, _ = self.discriminator((x_int1, x_int0.argmax(-1).float(), x_int0[:,:,:,1:].permute(0, 3, 1, 2)))
                discriminator_loss_w_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

                # train dscr real and fake losses
                discriminator_loss = real_discriminator_loss + fake_discriminator_loss + \
                                     self.exper_config.model_config["dscr"]["lambda_gp"] * discriminator_loss_w_gp
                self.discriminator_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

            if step % self.exper_config.log_every == 0 and step is not 0:
                print(step)
                print(type(step))
                print(generator_diffpool_losses[0].to("cpu").detach().numpy())
                print(generator_diffpool_losses[1].to("cpu").detach().numpy())

#               fake_discriminator_accuracy = self.accuracy_scores_(fake_label_var, fake_discriminator_preds)
#               real_discriminator_accuracy = self.accuracy_scores_(real_label_var, real_discriminator_preds)
#               generator_accuracy = self.accuracy_scores_(fake_label_var, generator_train_discriminator_preds)

                self.exper_config.summary_writer.add_scalars(
                    "{}/train/chem_metrics".format(self.exper_config.curr_exper_name_replica),
                    {"generator_valid": np.mean(generator_valid),
                     "generator_unique": np.mean(generator_unique),
                     "generator_novel": np.mean(generator_novel),
                     "generator_diverse": np.mean(generator_diverse),
                     "generator_drug_candidate_score": np.mean(generator_drug_candidate_score)},
                    int(step))
#               self.exper_config.summary_writer.add_scalars(
#                   "{}/train/accuracies".format(self.exper_config.curr_exper_name_replica),
#                   {"fake_discriminator_accuracy": fake_discriminator_accuracy,
#                    "real_discriminator_accuracy": real_discriminator_accuracy,
#                    "generator_accuracy": generator_accuracy},
#                   step)
                self.exper_config.summary_writer.add_scalars(
                    "{}/train/generator_losses".format(self.exper_config.curr_exper_name_replica),
                    {"fake_generator_loss": np.mean(fake_generator_loss.to("cpu").detach().numpy()),
                     "generator_loss": np.mean(generator_loss.to("cpu").detach().numpy()),
                     "generator_loss_value": np.mean(generator_loss_value.to("cpu").detach().numpy()),
                     "real_reward": np.mean(real_reward.to("cpu").detach().numpy()),
                     "fake_reward": np.mean(fake_reward.to("cpu").detach().numpy()),
                     "generator_diffpool_losses[0]": np.mean(generator_diffpool_losses[0].to("cpu").detach().numpy()),
                     "generator_diffpool_losses[1]": np.mean(generator_diffpool_losses[1].to("cpu").detach().numpy())},
                    int(step))
                self.exper_config.summary_writer.add_scalars(
                    "{}/train/real_discriminator_losses".format(self.exper_config.curr_exper_name_replica),
                    {"real_discriminator_loss": real_discriminator_loss.to("cpu").detach().numpy(),
                     "fake_discriminator_loss": fake_discriminator_loss.to("cpu").detach().numpy(),
                     "discriminator_loss_w_gp": discriminator_loss_w_gp.to("cpu").detach().numpy(),
                     "real_discriminator_diffpool_losses[0]": real_discriminator_diffpool_losses[0].to("cpu").detach().numpy(),
                     "real_discriminator_diffpool_losses[1]": real_discriminator_diffpool_losses[1].to("cpu").detach().numpy(),
                     "fake_discriminator_diffpool_losses[0]": fake_discriminator_diffpool_losses[0].to("cpu").detach().numpy(),
                     "fake_discriminator_diffpool_losses[1]": fake_discriminator_diffpool_losses[1].to("cpu").detach().numpy()},
                    int(step))
                for name, param in self.discriminator.named_parameters():
                    if param.requires_grad == True:
                        self.exper_config.summary_writer.add_histogram(
                            "{}/train/{}".format(self.exper_config.curr_exper_name_replica,
                                                 name),
                            param,
                            int(step))

            # Save model checkpoints.
            if (step + 1) % self.exper_config.chkpt_every == 0:
                G_path = os.path.join(self.exper_config.paths["EXPER_CHKPTS_DIR"], self.curr_exper_name_replica,
                                      '{}-G.ckpt'.format(step + 1))
                D_path = os.path.join(self.exper_config.paths["EXPER_CHKPTS_DIR"], self.curr_exper_name_replica,
                                      '{}-D.ckpt'.format(step + 1))
                V_path = os.path.join(self.exper_config.paths["EXPER_CHKPTS_DIR"], self.curr_exper_name_replica,
                                      '{}-V.ckpt'.format(step + 1))
                torch.save(self.generator.state_dict(), G_path)
                torch.save(self.discriminator.state_dict(), D_path)
                torch.save(self.value.state_dict(), V_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            if step % self.exper_config.validate_every == 0:
                print('val')
                # self.validate()

        # self.test()
