from sklearn.cluster import k_means
import torch
import cluster_clc


def _cluster(args, trainer,  test_loader):
    # train_theta_list, train_label_list = trainer.extract_theta(train_loader)
    test_theta_list, test_label_list = trainer.extract_theta(test_loader)

    # cluster_theta_list = train_theta_list + test_theta_list
    # cluster_theta_label = train_label_list + test_label_list

    cluster_theta_list = test_theta_list
    cluster_theta_label = test_label_list

    # # stand
    # cluster_theta_list = cluster_clc.standardization(cluster_theta_list)



    concat_cluster_theta = []
    layer_cluster_theta = [0.] * trainer.layer_num
    for layer_idx in range(trainer.layer_num):
        layer_cluster_theta[layer_idx] = np.concatenate([cluster_clc.standardization(theta_list[layer_idx]) for theta_list in cluster_theta_list], axis=1).transpose(1,0)
        concat_cluster_theta.append(np.concatenate([cluster_clc.standardization(theta_list[layer_idx]) for theta_list in cluster_theta_list], axis=1))

    concat_cluster_theta = np.concatenate(concat_cluster_theta, axis=0).transpose(1,0)

    cluster_label = np.concatenate(cluster_theta_label, axis=0)
    n_class = np.unique(cluster_label).size


    # concat_pred = KMeans(n_clusters=n_class, n_init=10, random_state=9, max_iter=10000).fit_predict(concat_cluster_theta)
    concat_cluster_theta[np.isnan(concat_cluster_theta)] = 0.
    concat_pred = k_means(concat_cluster_theta, n_class)[1]
    concat_purity_value = cluster_clc.purity(cluster_label, concat_pred)
    concat_nmi_value = cluster_clc.normalized_mutual_info_score(cluster_label, concat_pred)
    concat_ami_value = cluster_clc.adjusted_mutual_info_score(cluster_label, concat_pred)
    # concat_acc = Accuracy(cluster_label, concat_pred)
    print(f'CONCAT ==> purity:{concat_purity_value}, nmi_value:{concat_nmi_value}, ami_value:{concat_ami_value}')


    # concat_acc = Accuracy(cluster_label, concat_pred)
    # print(f'concat_acc: {concat_acc}')

    layer_pred = [0.] * trainer.layer_num
    layer_accs = [0.] * trainer.layer_num
    layer_purity = [0.] * trainer.layer_num
    layer_nmi = [0.] * trainer.layer_num
    layer_ami = [0.] * trainer.layer_num
    layer_acc = [0.] * trainer.layer_num
    for layer_idx in range(trainer.layer_num):
        # layer_pred[layer_idx] = KMeans(n_clusters=n_class, n_init=10, random_state=9, max_iter=10000).fit_predict(layer_cluster_theta[layer_idx])
        layer_pred[layer_idx] = k_means(layer_cluster_theta[layer_idx], n_class)[1]
        layer_purity[layer_idx] = cluster_clc.purity(cluster_label, layer_pred[layer_idx])
        layer_nmi[layer_idx] = cluster_clc.normalized_mutual_info_score(cluster_label, layer_pred[layer_idx])
        layer_ami[layer_idx] = cluster_clc.adjusted_mutual_info_score(cluster_label, layer_pred[layer_idx])
        # layer_acc[layer_idx] = Accuracy(cluster_label, layer_pred[layer_idx])
        # print(f'layer {layer_idx} acc: {layer_accs[layer_idx]}')
        print(f'layer {layer_idx} acc:{layer_acc[layer_idx]} purity: {layer_purity[layer_idx]}, nmi: {layer_nmi[layer_idx]}, ami: {layer_ami[layer_idx]}')

    return concat_purity_value, concat_nmi_value, layer_acc, layer_purity, layer_nmi

import tqdm
import utils
def _best_cluster(args, trainer, test_loader):
    # save best performance based on the concat_nmi
    best_concat_acc, best_concat_purity_value, best_concat_nmi_value = 0., 0., 0.
    best_layer_acc, best_layer_purity, best_layer_nmi = [0.] * trainer.layer_num, [0.] * trainer.layer_num, [0.] * trainer.layer_num
    best_epoch = 0

    # save best performance based on the layer_nmi[0]
    best_concat_acc_1, best_concat_purity_value_1, best_concat_nmi_value_1 = 0., 0., 0.
    best_layer_acc_1, best_layer_purity_1, best_layer_nmi_1 = [0.] * trainer.layer_num, [0.] * trainer.layer_num, [0.] * trainer.layer_num
    best_epoch_1 = 0

    for epoch in range(args.epochs):
        trainer.train_for_clustering(test_loader)

        concat_purity_value, concat_nmi_value, layer_acc, layer_purity, layer_nmi = _cluster(args, trainer, test_loader)
        if concat_nmi_value > best_concat_nmi_value:
            best_concat_purity_value, best_concat_nmi_value = concat_purity_value, concat_nmi_value
            best_layer_acc, best_layer_purity, best_layer_nmi = layer_acc, layer_purity, layer_nmi
            best_epoch = epoch
            log_str = f'epoch_based_on_CONCAT:{best_epoch}\nbest_concat_purity_value:{best_concat_purity_value}, best_concat_nmi_value:{best_concat_nmi_value}' + \
                f'best_layer_acc0:{best_layer_acc[0]}, best_layer_purity0:{best_layer_purity[0]}, best_layer_nmi:{best_layer_nmi[0]}'
            utils.add_show_log(trainer.log_path, log_str)
        if layer_nmi[0] > best_layer_nmi_1[0]:
            best_concat_purity_value_1, best_concat_nmi_value_1 = concat_purity_value, concat_nmi_value
            best_layer_purity_1, best_layer_nmi_1 = layer_purity, layer_nmi
            best_epoch_1 = epoch
            log_str = f'epoch_based_on_LAYER1:{best_epoch_1}\nbest_concat_purity_value_1:{best_concat_purity_value_1}, best_concat_nmi_value_1:{best_concat_nmi_value_1}' + \
                f'best_layer_acc0_1:{best_layer_acc_1[0]}, best_layer_purity0_1:{best_layer_purity_1[0]}, best_layer_nmi:{best_layer_nmi_1[0]}'
            utils.add_show_log(trainer.log_path, log_str)


import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def MutualInfo(L11, L22):
    # L11 is the groudtrue : (nsamples,)
    # L22 is the pre_label : (nsamples,)
    L1 = L11.copy()
    L2 = L22.copy()
    n_gnd = L1.shape[0]
    n_label = L2.shape[0]
    # assert n_gnd == n_label

    Label = np.unique(L1)
    nClass = len(Label)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    if nClass2 < nClass:
        L1 = np.concatenate((L1, Label))
        L2 = np.concatenate((L2, Label))
    else:
        L1 = np.concatenate((L1, Label2))
        L2 = np.concatenate((L2, Label2))

    G = np.zeros([nClass, nClass])
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == Label[i]) * (L2 == Label[j]))

    sum_G = np.sum(G)
    P1 = np.sum(G, axis=1)
    P1 = P1 / sum_G
    P2 = np.sum(G, 0)
    P2 = P2 / sum_G
    if np.sum((P1 == 0)) > 0 or np.sum((P2 == 0)):
        print('error ! Smooth fail !')
    else:
        H1 = np.sum(-P1 * np.log2(P1))
        H2 = np.sum(-P2 * np.log2(P2))
        P12 = G / sum_G
        PPP = P12 / np.tile(P2, (nClass, 1)) / np.tile(P1.reshape(-1, 1), (1, nClass))
        PPP[np.where(abs(PPP) < 1E-12)] = 1
        MI = np.sum(P12 * np.log2(PPP))
        MIhat = MI / np.max((H1, H2))

        return MIhat


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



# cluster_theta_norm = standardization(cluster_theta)
# train_n_class = np.unique(cluster_label).size
# tmp = k_means(cluster_theta_norm, train_n_class)
# predict_label = tmp[1]
# MHI = MutualInfo(cluster_label, predict_label)
# Ac = Accuracy(cluster_label, predict_label)
# if MHI > self.best_nmi:
#     self.best_nmi = MHI
#     self.best_acc = Ac
# print(
#     f'cluster : acc : {Ac:.4f}, nmi : {MHI:.4F}, best_acc: {self.best_acc:.4f}, best_nmi: {self.best_nmi:.4f}')





