import torch
import torch.nn as nn
from torch.utils import tensorboard
from torch.nn.parameter import Parameter

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model_dcetm import DCETM
from model_dcetm_beta import beta_DCETM
import utils


class DeepCoupling_Policy_trainer(object):
    def __init__(self, args, voc_path='voc.txt'):
        super(DeepCoupling_Policy_trainer, self).__init__()
        self.args = args
        self.save_path = args.save_path

        self.discount = args.discount
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)

        # model
        if args.use_beta:
            self.model = beta_DCETM(args)
            self.decoder_optimizer = torch.optim.Adam([{'params':self.model.decoder.parameters()},
                                                       {'params':self.model.betaDecoder.parameters()}], lr=self.lr,
                                                      weight_decay=self.weight_decay)
        else:
            self.model = DCETM(args)
            self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr,
                                                      weight_decay=self.weight_decay)

        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.layer_alpha_optimizer = torch.optim.Adam([self.model.layer_alpha], lr=self.lr * 0.01, weight_decay=self.weight_decay)

        # log
        self.writer = tensorboard.SummaryWriter(os.path.join(args.save_path, "tf_log"))
        self.log_path = os.path.join(args.save_path, "log.txt")
        utils.add_show_log(self.log_path, str(args))


    def train(self, train_data_loader, test_data_loader=None):
        for epoch in range(self.epochs):
            utils.add_show_log(self.log_path, "======================== train ========================")
            utils.add_show_log(self.log_path, f"epoch {epoch:03d}|{self.epochs:03d}:")

            self.model.to(self.args.device)
            KL_batch = [0] * (self.layer_num)
            likelihood_batch = [0] * (self.layer_num)
            division_likeli_loss_batch = 0.0
            num_data = len(train_data_loader)

            for train_data, train_label in tqdm.tqdm(train_data_loader):
                train_data = train_data.to(self.args.device)

                # ----------- RL training schema ---------------#
                for step in range(self.layer_num):
                    # update inference network
                    self.model.h_encoder.train()
                    self.model.shape_encoder.train()
                    self.model.scale_encoder.train()
                    self.model.decoder.eval()

                    ret_dict = self.model(train_data)
                    KL_loss = ret_dict['loss'][1:]
                    Likelihood = ret_dict['likelihood'][1:]

                    Q_value = torch.tensor(0., device=self.args.device)
                    for t in range(self.layer_num-1-step, -1, -1):  # from layer layer_num-1-step to 0
                        Q_value += pow(self.discount, self.layer_num-step-1 - t) * (Likelihood[t] + self.args.kl_weight*KL_loss[t])
                    Q_value.backward()

                    for para in self.model.parameters():
                        flag = torch.sum(torch.isnan(para))

                    if (flag == 0):
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # update generative network
                    self.model.h_encoder.eval()
                    self.model.shape_encoder.eval()
                    self.model.scale_encoder.eval()
                    self.model.decoder.train()

                    ret_dict = self.model(train_data)
                    KL_loss = ret_dict['loss'][1:]
                    Likelihood = ret_dict['likelihood'][1:]

                    Q_value = torch.tensor(0., device=self.args.device)
                    for t in range(self.layer_num-1-step, -1, -1):  # from layer layer_num-1-step to 0
                        Q_value += pow(self.discount, self.layer_num-step-1 - t) * (Likelihood[t] + self.args.kl_weight*KL_loss[t])
                    Q_value.backward()

                    for para in self.model.parameters():
                        flag = torch.sum(torch.isnan(para))

                    if (flag == 0):
                        nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                        self.decoder_optimizer.step()
                        self.decoder_optimizer.zero_grad()

                # update alpha
                self.model.train()  # require_grad for alpha
                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.eval()

                ret_dict = self.model(train_data)

                division_loss = ret_dict["division_likeli_loss"]
                division_loss.backward()
                self.layer_alpha_optimizer.step()
                self.layer_alpha_optimizer.zero_grad()

                division_likeli_loss_batch += ret_dict['division_likeli_loss'].item() / num_data

                for t in range(self.layer_num):
                    KL_batch[t] += ret_dict['loss'][t + 1].item() / num_data
                    likelihood_batch[t] += ret_dict['likelihood'][t + 1].item() / num_data


            # evaluate
            if epoch % self.args.eval_epoch_num == 0:
                # write to the file
                activated_layer_alpha = torch.nn.functional.sigmoid(self.model.layer_alpha)
                softmax_layer_alpha = torch.nn.functional.softmax(activated_layer_alpha, dim=0)
                print(f'softmax_layer_alpha: {softmax_layer_alpha}')
                for t in range(self.layer_num):
                    log_str = 'epoch {}|{}, layer {}|{}, kl: {}, likelihood: {}, devision_likelihood: {}'.format(epoch, self.epochs, t,
                                                                                              self.layer_num,
                                                                                              KL_batch[t],
                                                                                              likelihood_batch[t],
                                                                                              division_likeli_loss_batch)
                    utils.add_show_log(self.log_path, log_str)

                    self.writer.add_scalar(f"train/kl_loss{t}", KL_batch[t], epoch)
                    self.writer.add_scalar(f"train/likelihood_{t}", likelihood_batch[t], epoch)
                self.writer.add_scalar(f"train/division_likelihood", division_likeli_loss_batch, epoch)


            if epoch % self.args.test_epoch_num == 0  and self.args.ppl:
                if self.args.clustering:
                    pass
                else:
                    self.test(train_data_loader, epoch)

            if (epoch+1) % 100 == 0:
                model_path = os.path.join(self.args.save_path, "model")
                file_name = model_path + '/dcetm_policy_' + self.args.task + str(epoch+1) + '.pt'
                utils.save_checkpoint({'state_dict': self.model.state_dict()}, file_name, True)

    def test(self, data_loader, epoch):
        utils.add_show_log(self.log_path, "======================== test ========================")

        # test
        self.model.eval()
        num_data = len(data_loader)
        KL_batch = [0] * (self.layer_num)
        PPL_batch = [0] * (self.layer_num)
        likelihood_batch = [0] * (self.layer_num)
        division_PPL_batch = 0.0
        division_likeli_loss_batch = 0.0

        for test_data, test_label in tqdm.tqdm(data_loader):
            test_data = test_data.to(self.args.device)
            test_label = test_label.to(self.args.device)

            ret_dict = self.model.test_ppl(test_data, test_label)
            PPL_minibatch = ret_dict["ppl"]
            division_PPL_batch += ret_dict["ppl"][0].item() / num_data

            ret_dict = self.model.forward(test_label)
            division_likeli_loss_batch += ret_dict["division_likeli_loss"].item() / num_data

            for t in range(self.layer_num):
                KL_batch[t] += ret_dict['loss'][t + 1].item() / num_data
                likelihood_batch[t] += ret_dict['likelihood'][t + 1].item() / num_data
                PPL_batch[t] += PPL_minibatch[t + 1].item() / num_data

        # write to the file
        for t in range(self.layer_num):
            log_str = 'epoch {}|{}, layer {}|{}, ppl:{}, devision_ppl: {}, likelihood: {}, devision_likelihood: {}'.format(epoch, self.epochs,
                                                                                                                 t, self.layer_num,
                                                                                                                 PPL_batch[t],
                                                                                                                 division_PPL_batch,
                                                                                                         likelihood_batch[t],
                                                                                                         division_likeli_loss_batch)
            utils.add_show_log(self.log_path, log_str)

            self.writer.add_scalar(f"test/kl_loss{t}", KL_batch[t], epoch)
            self.writer.add_scalar(f"test/likelihood_{t}", likelihood_batch[t], epoch)
            self.writer.add_scalar(f"test/PPL_{t}", PPL_batch[t], epoch)
        self.writer.add_scalar(f"test/devision_ppl", division_PPL_batch, epoch)
        self.writer.add_scalar(f"test/division_likelihood", division_likeli_loss_batch, epoch)

    def train_for_clustering(self, train_data_loader):
        self.model.to(self.args.device)
        KL_batch = [0] * (self.layer_num)
        likelihood_batch = [0] * (self.layer_num)
        division_likeli_loss_batch = 0.0
        num_data = len(train_data_loader)

        for train_data, train_label in tqdm.tqdm(train_data_loader):
            train_data = train_data.to(self.args.device)

            # ----------- RL training schema ---------------#
            for step in range(self.layer_num):
                # update inference network
                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                ret_dict = self.model(train_data)
                KL_loss = ret_dict['loss'][1:]
                Likelihood = ret_dict['likelihood'][1:]

                Q_value = torch.tensor(0., device=self.args.device)
                for t in range(self.layer_num - 1 - step, -1, -1):  # from layer layer_num-1-step to 0
                    Q_value += pow(self.discount, self.layer_num - step - 1 - t) * (
                                Likelihood[t] + self.args.kl_weight * KL_loss[t])
                Q_value.backward()

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # update generative network
                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                ret_dict = self.model(train_data)
                KL_loss = ret_dict['loss'][1:]
                Likelihood = ret_dict['likelihood'][1:]

                Q_value = torch.tensor(0., device=self.args.device)
                for t in range(self.layer_num - 1 - step, -1, -1):  # from layer layer_num-1-step to 0
                    Q_value += pow(self.discount, self.layer_num - step - 1 - t) * (
                                Likelihood[t] + self.args.kl_weight * KL_loss[t])
                Q_value.backward()

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()

            # update alpha
            self.model.train()  # require_grad for alpha
            self.model.h_encoder.eval()
            self.model.shape_encoder.eval()
            self.model.scale_encoder.eval()
            self.model.decoder.eval()

            ret_dict = self.model(train_data)

            division_loss = ret_dict["division_likeli_loss"]
            division_loss.backward()
            self.layer_alpha_optimizer.step()
            self.layer_alpha_optimizer.zero_grad()

            division_likeli_loss_batch += ret_dict['division_likeli_loss'].item() / num_data

            for t in range(self.layer_num):
                KL_batch[t] += ret_dict['loss'][t + 1].item() / num_data
                likelihood_batch[t] += ret_dict['likelihood'][t + 1].item() / num_data

    def load_model(self, save_path):
        checkpoint = torch.load(save_path, map_location=self.args.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.args.device)

    def extract_theta(self, data_loader):
        self.model.eval()
        test_theta_batch = []
        test_label_batch = []
        for test_data, test_label in tqdm.tqdm(data_loader):
            test_data = test_data.type(torch.float).to(self.args.device)
            test_label = test_label.to(self.args.device)
            ret_dict = self.model.forward(test_data)
            test_theta_batch.append([ret_dict['theta'][i].detach().cpu().numpy() for i in range(self.layer_num)])
            test_label_batch.append(test_label.detach().cpu().numpy())

        return test_theta_batch, test_label_batch

    def vis(self):
        # layer1
        w_1 = torch.mm(self.GBN_models[0].decoder.rho, torch.transpose(self.GBN_models[0].decoder.alphas, 0, 1))
        phi_1 = torch.softmax(w_1, dim=0).cpu().detach().numpy()

        index1 = range(100)
        dic1 = phi_1[:, index1[0:49]]
        # dic1 = phi_1[:, :]
        fig7 = plt.figure(figsize=(10, 10))
        for i in range(dic1.shape[1]):
            tmp = dic1[:, i].reshape(28, 28)
            ax = fig7.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index1[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_2 = torch.mm(self.GBN_models[1].decoder.rho, torch.transpose(self.GBN_models[1].decoder.alphas, 0, 1))
        phi_2 = torch.softmax(w_2, dim=0).cpu().detach().numpy()
        index2 = range(49)
        dic2 = np.matmul(phi_1, phi_2[:, index2[0:49]])
        # dic2 = np.matmul(phi_1, phi_2[:, :])
        fig8 = plt.figure(figsize=(10, 10))
        for i in range(dic2.shape[1]):
            tmp = dic2[:, i].reshape(28, 28)
            ax = fig8.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index2[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_3 = torch.mm(self.GBN_models[2].decoder.rho, torch.transpose(self.GBN_models[2].decoder.alphas, 0, 1))
        phi_3 = torch.softmax(w_3, dim=0).cpu().detach().numpy()
        index3 = range(32)

        dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, index3[0:32]])
        # dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, :])
        fig9 = plt.figure(figsize=(10, 10))
        for i in range(dic3.shape[1]):
            tmp = dic3[:, i].reshape(28, 28)
            ax = fig9.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index3[i] + 1))
            ax.imshow(tmp)

        plt.show()

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath, top_n=50):
        if self.voc is not None:
            utils.chk_mkdir(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

    def vis_txt(self, epoch):
        phi = []
        for t in range(self.layer_num):
            phi.append(self.model.decoder[t].get_phi().cpu().detach().numpy())

        if epoch % self.args.save_phi_every_epoch == 0:
            self.vision_phi(phi, os.path.join(self.save_path, "phi", f"{epoch:03d}"))
