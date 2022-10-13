import os
from graphviz import Digraph

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function

real_min = torch.tensor(1e-30)

def chk_mkdir(paths):
    paths = paths.split('/')
    root = ''
    for path in paths:
        root = os.path.join(root, path)
        if not os.path.exists(root):
            os.makedirs(root)

def add_show_log(path, string, mode='a'):
    with open(path, mode) as f:
        f.write(string + "\n")
        print(string)

def who_need_grad(model):
    # need_grad = []
    for p in model.parameters():
        if p.grad == None:
            p.requires_grad = False
    # return need_grad

def restore_grad(model):
    for p in model.parameters():
        p.requires_grad = True

def clear_nan_grad(model):
    for p in model.parameters():
        p.grad = p.grad.where(~torch.isnan(p.grad), torch.tensor(0., device=p.grad.device))

def clear_rl_nan_grad(model, step, layer_num):
    # for shape_encoder, scale_encoder
    for i in range(step+1):
        idx = layer_num - 1 - i
        clear_nan_grad(model[idx])

def set_model_train(model, flag=True):
    model.train(flag)
    for p in model.parameters():
        p.requires_grad = flag


def log_max(x):
    return torch.log(torch.max(x, real_min.to(x.device)))


def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):
    eulergamma = torch.tensor(0.5772, dtype=torch.float32)

    part1 = eulergamma.cuda() * (1 - 1 / Wei_shape) + log_max(
        Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)

    part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma.cuda() / Wei_shape)

    part3 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))

    KL = part1 + part2 + part3
    return KL


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)

            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class DeepConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(DeepConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf)
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf))

            w2 = torch.empty(nf, nf)
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf))

        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            rx = x
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = x + rx
        else:
            raise NotImplementedError
        return x


class ResConv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(ResConv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w1 = torch.empty(nx, nf)
            nn.init.normal_(w1, std=0.02)
            self.w1 = Parameter(w1)
            self.b1 = Parameter(torch.zeros(nf))

            w2 = torch.empty(nf, nf)
            nn.init.normal_(w2, std=0.02)
            self.w2 = Parameter(w2)
            self.b2 = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            rx = x
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b1, x.view(-1, x.size(-1)), self.w1)
            x = torch.nn.functional.relu(x)
            x = torch.addmm(self.b2, x.view(-1, x.size(-1)), self.w2)
            x = x.view(*size_out)
            x = rx + x
        else:
            raise NotImplementedError
        return x


# class Conv1DSoftmax(nn.Module):
#     def __init__(self, voc_size, topic_size):
#         super(Conv1DSoftmax, self).__init__()
#
#         w = torch.empty(voc_size, topic_size)
#         nn.init.normal_(w, std=0.02)
#         self.w = Parameter(w)
#
#     def forward(self, x):
#         w = torch.softmax(self.w, dim=0)
#         x = torch.mm(w, x.view(-1, x.size(-1)))
#         return x


class Conv1DSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size, last_layer=None):
        super(Conv1DSoftmaxEtm, self).__init__()
        self.voc_size = voc_size
        self.topic_size = topic_size
        self.emb_size = emb_size

        if last_layer is None:
            w1 = torch.empty(self.voc_size, self.emb_size)
            nn.init.normal_(w1, std=0.02)
            self.rho = Parameter(w1)
        else:
            w1 = torch.empty(self.voc_size, self.emb_size)
            nn.init.normal_(w1, std=0.02)
            # self.rho = last_layer.alphas
            self.rho = Parameter(w1)

        w2 = torch.empty(self.topic_size, self.emb_size)
        nn.init.normal_(w2, std=0.02)
        self.alphas = Parameter(w2)

    def get_phi(self):
        w_t = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        return torch.softmax(w_t, dim=0)

    def forward(self, x, t):
        if t == 0:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        else:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
            # w = torch.mm(self.rho, torch.transpose(self.alphas.detach(), 0, 1))

        w = torch.softmax(w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

class betaSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size, beta0, last_layer=None):
        super(betaSoftmaxEtm, self).__init__()
        self.voc_size = voc_size
        self.topic_size = topic_size
        self.emb_size = emb_size

        # if last_layer is None:
        #     w1 = torch.empty(self.voc_size, self.emb_size)
        #     nn.init.normal_(w1, std=0.02)
        #     self.beta0 = Parameter(w1)
        # else:
        #     w1 = torch.empty(self.voc_size, self.emb_size)
        #     nn.init.normal_(w1, std=0.02)
        #     # self.rho = last_layer.alphas
        #     self.beta0 = Parameter(w1)
        self.beta0 = beta0
        w2 = torch.empty(self.topic_size, self.emb_size)
        nn.init.normal_(w2, std=0.02)
        self.beta_t = Parameter(w2)

    def get_phi(self):
        w_t = torch.mm(self.beta0, torch.transpose(self.beta_t, 0, 1))
        return torch.softmax(w_t, dim=0)

    def forward(self, x, t):
        if t == 0:
            w = torch.mm(self.beta0, torch.transpose(self.beta_t, 0, 1))
        else:
            w = torch.mm(self.beta0, torch.transpose(self.beta_t, 0, 1))

        w = torch.softmax(w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x

def variable_para(shape, device='cuda'):
    w = torch.empty(shape, device=device)
    nn.init.normal_(w, std=0.02)
    return torch.tensor(w, requires_grad=True)


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")

def list_add(l1, l2):
    c = []
    for i in range(len(l1)):
        c.append(l1[i]+l2[i])
    return c

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        try:
            grad_output = grad_output.data
        except:
            print('Fail to get grad')
            return True
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                # assert fn in fn_dict, fn_dict
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

if __name__ == '__main__':
    x = Variable(torch.randn(10, 10), requires_grad=True)
    y = Variable(torch.randn(10, 10), requires_grad=True)

    z = x / y
    m = z.sum() * 2
    get_dot = register_hooks(m)
    m.backward()
    dot = get_dot()
    dot.save('tmp.dot')