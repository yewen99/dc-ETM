import torch.nn.functional as F
from utils import *


class DCETM(nn.Module):
    def __init__(self, args):
        super(DCETM, self).__init__()
        self.args = args

        self.real_min = torch.tensor(1e-30, device=args.device)
        self.wei_shape_max = torch.tensor(10.0, device=args.device)
        self.wei_shape = torch.tensor(1e-1, device=args.device)

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.topic_size = args.topic_size
        self.embed_size = args.embed_size
        self.topic_size = [self.vocab_size] + self.topic_size
        self.layer_num = len(self.topic_size) - 1

        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for i in range(self.layer_num+1)])

        h_encoder = [DeepConv1D(self.hidden_size, 1, self.vocab_size)]
        for i in range(self.layer_num - 1):
            h_encoder.append(ResConv1D(self.hidden_size, 1, self.hidden_size))
        self.h_encoder = nn.ModuleList(h_encoder)

        shape_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in range(self.layer_num - 1)]
        shape_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.shape_encoder = nn.ModuleList(shape_encoder)

        scale_encoder = [Conv1D(self.topic_size[i + 1], 1, self.topic_size[i + 1] + self.hidden_size) for i in range(self.layer_num - 1)]
        scale_encoder.append(Conv1D(self.topic_size[self.layer_num], 1, self.hidden_size))
        self.scale_encoder = nn.ModuleList(scale_encoder)

        decoder = [Conv1DSoftmaxEtm(self.topic_size[i], self.topic_size[i + 1], self.embed_size) for i in range(self.layer_num)]
        self.decoder = nn.ModuleList(decoder)
        for t in range(self.layer_num - 1):
            self.decoder[t + 1].rho = self.decoder[t].alphas

        self.layer_alpha = Parameter(torch.ones(self.layer_num)/self.layer_num)

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min))

    def reparameterize(self, Wei_shape_res, Wei_scale, Sample_num = 50):
        # sample one
        eps = torch.FloatTensor(Sample_num, Wei_shape_res.shape[0], Wei_shape_res.shape[1]).uniform_(0, 1).to(self.args.device)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(Sample_num, 1, 1) \
                * torch.pow(-self.log_max(1 - eps),  torch.unsqueeze(Wei_shape_res, axis=0).repeat(Sample_num, 1, 1))  #
        return torch.mean(theta, dim=0, keepdim=False)

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1))
        return - likelihood / x.shape[1]

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        eulergamma = torch.tensor(0.5772, device=self.args.device)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma * Gam_shape * Wei_shape_res + self.log_max(Wei_shape_res)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / Wei_scale.shape[1]

    def _ppl(self, x, phi_theta):

        PPLs = [0.0] * (self.layer_num + 1)
        re_x_all_1 = 0.0

        activated_layer_alpha = torch.nn.functional.sigmoid(self.layer_alpha)
        softmax_layer_alpha = torch.nn.functional.softmax(activated_layer_alpha, dim=0)

        for t in range(self.layer_num + 1):
            if t == 0:
                pass
            else:
                phi_theta_tmp = phi_theta[t - 1]
                for j in range(t - 1):
                    phi_theta_tmp = torch.mm(self.decoder[t - j - 2].get_phi(), phi_theta_tmp.view(-1, phi_theta_tmp.size(-1)))
                re_x1 = phi_theta_tmp
                re_x2 = re_x1 / (re_x1.sum(0) + real_min)
                ppl = x * torch.log(re_x2.T + real_min) / -x.sum()
                PPLs[t] = ppl.sum().exp()

                # total reconstruct
                re_x_all_1 += softmax_layer_alpha[t-1] * phi_theta_tmp.detach()

        re_x_all_2 = re_x_all_1 / (re_x_all_1.sum(0) + real_min)
        PPLs[0] = (x * torch.log(re_x_all_2.T + real_min) / -x.sum()).sum().exp()

        return PPLs

    def test_ppl(self, x, y):
        ret_dict = self.forward(x)
        PPLs = self._ppl(y, ret_dict["phi_theta"])
        ret_dict.update({"ppl": PPLs})

        return ret_dict

    def forward(self, x):
        hidden_list = [0] * self.layer_num
        theta = [0] * self.layer_num
        k_rec = [0] * self.layer_num
        l = [0] * self.layer_num
        l_tmp = [0] * self.layer_num
        phi_theta = [0] * self.layer_num

        loss = [0] * (self.layer_num + 1)
        likelihood = [0] * (self.layer_num + 1)

        for t in range(self.layer_num):
            if t == 0:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](x)))
            else:
                hidden = F.relu(self.bn_layer[t](self.h_encoder[t](hidden_list[t-1])))
            hidden_list[t] = hidden

        for t in range(self.layer_num-1, -1, -1):
            if t == self.layer_num - 1:
                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_list[t])), self.real_min.to(self.args.device))      # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.to(self.args.device))

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_list[t])), self.real_min.to(self.args.device))
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                phi_theta[t] = self.decoder[t](theta[t], t)

            else:
                temp = phi_theta[t+1].permute(1, 0)
                hidden_phitheta = torch.cat((hidden_list[t], temp), 1)

                k_rec_temp = torch.max(torch.nn.functional.softplus(self.shape_encoder[t](hidden_phitheta)), self.real_min.to(self.args.device))  # k_rec = 1/k
                k_rec[t] = torch.min(k_rec_temp, self.wei_shape_max.to(self.args.device))

                l_tmp[t] = torch.max(torch.nn.functional.softplus(self.scale_encoder[t](hidden_phitheta)), self.real_min.to(self.args.device))
                l[t] = l_tmp[t] / torch.exp(torch.lgamma(1 + k_rec[t]))

                theta[t] = self.reparameterize(k_rec[t].permute(1, 0), l[t].permute(1, 0))
                phi_theta[t] = self.decoder[t](theta[t], t)

        # calculate loss
        for t in range(self.layer_num + 1):
            if t == 0:
                loss[t] = self.compute_loss(x.permute(1, 0), phi_theta[t])
            elif t == self.layer_num:
                loss[t] = self.KL_GamWei(torch.tensor(0.1, device=self.args.device), torch.tensor(1.0, device=self.args.device), k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))
            else:
                loss[t] = self.KL_GamWei(phi_theta[t], torch.tensor(1.0, device=self.args.device), k_rec[t - 1].permute(1, 0), l[t - 1].permute(1, 0))

        phi_theta_tmp = [0.] * self.layer_num
        for t in range(self.layer_num):
            phi_theta_tmp[t] = phi_theta[t]
            for j in range(t):
                phi_theta_tmp[t] = torch.mm(self.decoder[t-j-1].get_phi(), phi_theta_tmp[t].view(-1, phi_theta_tmp[t].size(-1)))
            # calculate likelihood at the bottom layer
            likelihood[t+1] = self.compute_loss(x.permute(1, 0), phi_theta_tmp[t])

        activated_layer_alpha = torch.nn.functional.sigmoid(self.layer_alpha)
        softmax_layer_alpha = torch.nn.functional.softmax(activated_layer_alpha, dim=0)
        add_phi_theta = 0.
        for t in range(self.layer_num):
            # add_phi_theta += softmax_layer_alpha[t] * phi_theta_tmp[t].detach() # modified by chaojie 2022/1/3
            # add_phi_theta += (1/(self.layer_num)) * phi_theta_tmp[t]
            add_phi_theta += softmax_layer_alpha[t] * phi_theta_tmp[t]
        division_likeli_loss = self.compute_loss(x.permute(1, 0), add_phi_theta)
        likelihood[0] = division_likeli_loss

        return {"phi_theta": phi_theta,
                "theta": theta,
                "loss": loss,
                "likelihood": likelihood,
                "division_likeli_loss": division_likeli_loss}
