import pickle
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader

# minist
class CustomDataset(Dataset):
    def __init__(self, data_file):
        # with open(data_file, 'rb') as f:
        #     data = pickle.load(f)
        # self.train_data = data['doc_bow'].toarray()
        # self.N, self.vocab_size = self.train_data.shape
        # self.voc = data['word2id']
        data = sio.loadmat('mnist_data/mnist')
        self.train_data = np.array(np.ceil(data['train_mnist'] * 5), order='C')  # 0-1
        self.test_data = np.array(np.ceil(data['test_mnist'] * 5), order='C')  # 0-1
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N


def get_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomDataset(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size

# txt data
class CustomDataset_txt(Dataset):
    def __init__(self, data_file, voc_size):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        if voc_size == 2000:
            data_all = data['data_2000'].toarray()
            self.voc = data['voc2000']
        else:
            data_all = data['data_36804'].toarray()
            self.voc = data['voc36804']

        self.train_data = data_all[data['train_id']].astype("int32")
        self.train_label = [data['label'][i] for i in data['train_id']]
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        label_data = self.train_label[index]
        return torch.from_numpy(np.squeeze(topic_data)).float(), torch.tensor(label_data).float()

    def __len__(self):
        return self.N


class CustomTestDataset_txt(Dataset):
    def __init__(self, data_file, voc_size):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        if voc_size == 2000:
            data_all = data['data_2000'].toarray()
            self.voc = data['voc2000']
        elif voc_size == 36804:
            data_all = data['data_36804'].toarray()
            self.voc = data['voc36804']
        else:
            data_all = data['data_20000'].toarray()
            self.voc = data['voc20000']
        self.test_data = data_all[data['test_id']].astype("int32")
        self.test_label = [data['label'][i] for i in data['test_id']]
        self.N, self.vocab_size = self.test_data.shape

    def __getitem__(self, index):
        topic_data = self.test_data[index, :]
        label_data = self.test_label[index]
        return torch.from_numpy(np.squeeze(topic_data)).float(), torch.tensor(label_data).float()

    def __len__(self):
        return self.N

class CustomTest_ALL_Dataset_txt(Dataset):
    def __init__(self, data_file, voc_size):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        if voc_size == 2000:
            data_all = data['data_2000'].toarray()
            self.voc = data['voc2000']
        elif voc_size == 36804:
            data_all = data['data_36804'].toarray()
            self.voc = data['voc36804']
        else:
            data_all = data['data_20000'].toarray()
            self.voc = data['voc20000']
        self.train_data = data_all.astype("int32")
        self.train_label = data['label']
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        label_data = self.train_label[index]
        return torch.from_numpy(np.squeeze(topic_data)).float(), torch.tensor(label_data).float()

    def __len__(self):
        return self.N

class CustomTestDataset_r8_txt(Dataset):
    def __init__(self, data_file, voc_size):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        data_all = data['test_bow'].toarray()
        self.voc = data['voc']

        self.test_data = data_all.astype("int32")
        self.test_label = data['test_label']
        self.N, self.vocab_size = self.test_data.shape

    def __getitem__(self, index):
        topic_data = self.test_data[index, :]
        label_data = self.test_label[index]
        return torch.from_numpy(np.squeeze(topic_data)).float(), torch.tensor(label_data).float()

    def __len__(self):
        return self.N

class CustomTrainDataset_r8_txt(Dataset):
    def __init__(self, data_file, voc_size):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        data_all = data['train_bow'].toarray()
        self.voc = data['voc']

        self.test_data = data_all.astype("int32")
        self.test_label = data['train_label']
        self.N, self.vocab_size = self.test_data.shape

    def __getitem__(self, index):
        topic_data = self.test_data[index, :]
        label_data = self.test_label[index]
        return torch.from_numpy(np.squeeze(topic_data)).float(), torch.tensor(label_data).float()

    def __len__(self):
        return self.N

class CustomDataset_txt_ppl(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_36804'].toarray()
        self.train_data, self.test_data = gen_ppl_doc(data_all.astype("int32"))
        self.voc = data['voc36804']
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        return torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

class CustomDataset_txt_ppl_2000(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.train_data, self.test_data = gen_ppl_doc(data_all.astype("int32"))
        self.voc = data['voc2000']
        self.N, self.vocab_size = self.train_data.shape
        self.label = data['label']

    def __getitem__(self, index):
        return torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

class CustomDataset_txt_cluster_r8(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['fea']
        self.train_data = data_all.astype("int32")
        self.voc = data['voc']
        self.N, self.vocab_size = self.train_data.shape
        self.train_label = data['gnd']

    def __getitem__(self, index):
        return torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(np.squeeze(self.train_label[index])).float()

    def __len__(self):
        return self.N

class CustomDataset_txt_ppl_r8(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['fea']
        self.train_data, self.test_data = gen_ppl_doc(data_all.astype("int32"))
        self.voc = data['voc']
        self.N, self.vocab_size = self.train_data.shape
        self.train_label = data['gnd']

    def __getitem__(self, index):
        return torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

class CustomDataset_cluster_trec_6(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['fea']
        # self.train_data, self.test_data = gen_ppl_doc(data_all.astype("int32"))
        self.train_data = data_all.astype("int32")
        self.voc = data['voc']
        self.N, self.vocab_size = self.train_data.shape
        self.train_label = data['gnd']

    def __getitem__(self, index):
        return torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(
            np.squeeze(self.train_label[index])).float()

    def __len__(self):
        return self.N

class CustomDataset_txt_ppl_rcv1_2000(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.data_all = data['rcv2_bow_2000']
        self.voc = data['rcv2_voc_2000']
        del data
        self.N, self.vocab_size = self.data_all.shape

    def __getitem__(self, index):
        ret = self.data_all[index].toarray()
        train_data, test_data = gen_ppl_doc(ret.astype("int32"))
        return torch.from_numpy(np.squeeze(train_data)).float(), torch.from_numpy(np.squeeze(test_data)).float()

    def __len__(self):
        return self.N

class CustomDataset_txt_ppl_rcv1_10000(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.data_all = data['rcv2_bow_10000']
        self.voc = data['rcv2_voc_10000']
        del data
        self.N, self.vocab_size = self.data_all.shape

    def __getitem__(self, index):
        ret = self.data_all[index].toarray()
        train_data, test_data = gen_ppl_doc(ret.astype("int32"))
        return torch.from_numpy(np.squeeze(train_data)).float(), torch.from_numpy(np.squeeze(test_data)).float()

    def __len__(self):
        return self.N


def gen_ppl_doc(x, ratio=0.8):
    """
    inputs:
        x: N x V, np array,
        ratio: float or double,
    returns:
        x_1: N x V, np array, the first half docs whose length equals to ratio * doc length,
        x_2: N x V, np array, the second half docs whose length equals to (1 - ratio) * doc length,
    """
    import random
    x_1, x_2 = np.zeros_like(x), np.zeros_like(x)
    # indices_x, indices_y = np.nonzero(x)
    for doc_idx, doc in enumerate(x):
        indices_y = np.nonzero(doc)[0]
        l = []
        for i in range(len(indices_y)):
            value = doc[indices_y[i]]
            for _ in range(int(value)):
                l.append(indices_y[i])
        random.seed(2020)
        random.shuffle(l)
        l_1 = l[:int(len(l) * ratio)]
        l_2 = l[int(len(l) * ratio):]
        for l1_value in l_1:
            x_1[doc_idx][l1_value] += 1
        for l2_value in l_2:
            x_2[doc_idx][l2_value] += 1
    return x_1, x_2


def get_loader_txt(topic_data_file, batch_size=200, voc_size=36804, shuffle=True, num_workers=0):
    if topic_data_file[-13:] == 'rcv1_2000.pkl':
        dataset = CustomDataset_txt_ppl_rcv1_2000(topic_data_file)

    if topic_data_file[-12:] == 'rcv1_new.pkl':
        dataset = CustomDataset_txt_ppl_rcv1_10000(topic_data_file)

    if topic_data_file[-6:] == 'r8.pkl':
        dataset = CustomDataset_txt_cluster_r8(topic_data_file)

    if topic_data_file[-13:] == 'r8_little.pkl':
        dataset = CustomTrainDataset_r8_txt(topic_data_file, voc_size)

    if topic_data_file[-6:] == 'ng.pkl':
        dataset = CustomDataset_txt(topic_data_file, voc_size)

    # dataset = CustomDataset_txt(topic_data_file, voc_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), voc_size, dataset.voc

def get_test_loader_txt(topic_data_file, batch_size=200, voc_size=36804, shuffle=True, num_workers=0):
    if topic_data_file[-6:] == 'ng.pkl':
        # dataset = CustomTest_ALL_Dataset_txt(topic_data_file, voc_size)
        dataset = CustomTestDataset_txt(topic_data_file, voc_size)

    if topic_data_file[-13:] == 'r8_little.pkl':
        dataset = CustomTestDataset_r8_txt(topic_data_file, voc_size)
    if topic_data_file[-6:] == 'r8.pkl':
        dataset = CustomDataset_cluster_trec_6(topic_data_file)
    if topic_data_file[-10:] == 'trec_6.pkl' or topic_data_file[-14:] == 'trec_train.pkl' or topic_data_file[-13:] == 'trec_test.pkl'\
            or topic_data_file[-15:] == 'WebKB_train.pkl' or topic_data_file[-14:] == 'WebKB_test.pkl':
        dataset = CustomDataset_cluster_trec_6(topic_data_file)
    # if topic_data_file[-10:] == 'trec_train.pkl'
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), voc_size, dataset.voc


def get_loader_txt_ppl(topic_data_file, batch_size=200, voc_size=36804, shuffle=True, num_workers=4):
    if topic_data_file[-13:] == 'rcv1_2000.pkl':
        dataset = CustomDataset_txt_ppl_rcv1_2000(topic_data_file)

    if topic_data_file[-14:] == 'rcv1_10000.pkl':
        dataset = CustomDataset_txt_ppl_rcv1_10000(topic_data_file)

    if topic_data_file[-6:] == 'r8.pkl':
        dataset = CustomDataset_txt_ppl_r8(topic_data_file)

    if topic_data_file[-6:] == 'ng.pkl':
        if voc_size == 36804:
            dataset = CustomDataset_txt_ppl(topic_data_file)
        else:
            dataset = CustomDataset_txt_ppl_2000(topic_data_file)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc








# class CustomDataset(Dataset):
#     def __init__(self, data_file):
#         # with open(data_file, 'rb') as f:
#         #     data = pickle.load(f)
#         # self.train_data = data['doc_bow'].toarray()
#         # self.N, self.vocab_size = self.train_data.shape
#         # self.voc = data['word2id']
#         data = sio.loadmat('mnist_data/mnist')
#         self.train_data = np.array(np.ceil(data['train_mnist'] * 5), order='C')  # 0-1
#         self.test_data = np.array(np.ceil(data['test_mnist'] * 5), order='C')  # 0-1
#         self.N, self.vocab_size = self.train_data.shape
#
#     def __getitem__(self, index):
#         topic_data = self.train_data[index, :]
#         return np.squeeze(topic_data), 1
#
#     def __len__(self):
#         return self.N
#
# def get_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
#     dataset = CustomDataset(topic_data_file)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
#                       drop_last=True), dataset.vocab_size
#


# class CustomDataset(Dataset):
#     def __init__(self, data_file):
#         with open(data_file, 'rb') as f:
#             data = pickle.load(f)
#         train_id = data['train_id']
#         test_id = data['test_id']
#         train_data = data['data_2000']
#         test_data = data['data_2000'][test_id]
#         train_label = np.array(data['label'])[train_id]
#         test_label = np.array(data['label'])[test_id]
#         voc = data['voc2000']
#         self.train_data = train_data
#         self.N, self.vocab_size = self.train_data.shape
#         self.voc = voc
#
#     def __getitem__(self, index):
#         topic_data = self.train_data[index].toarray()
#         return np.squeeze(topic_data), 1
#
#     def __len__(self):
#         return self.N


# def get_loader_txt_ppl_withLabel(topic_data_file, train=True, batch_size=200, voc_size=36804, shuffle=True, num_workers=0):
#     if topic_data_file[-7:] == 'bow.pkl':
#         dataset = CustomDataset_txt_ppl_rcv1(topic_data_file)
#     if topic_data_file[-6:] == 'ng.pkl':
#         if voc_size == 36804:
#             dataset = CustomDataset_txt_ppl(topic_data_file)
#         else:
#             dataset = CustomDataset_ppl_ng2000_withLabel(topic_data_file, train)
#
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
#                       drop_last=True), dataset.vocab_size, dataset.voc

class CustomDataset_ppl_ng2000_withLabel(Dataset):
    def __init__(self, data_file, train=True):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.train_data, self.test_data = gen_ppl_doc(data_all.astype("int32"))
        total_len, _ = self.train_data.shape
        self.spilit_sapce = int(total_len * 0.8)
        self.train_classfication = self.train_data[:self.spilit_sapce]
        self.test_classfication = self.train_data[self.spilit_sapce:]
        self.voc = data['voc2000']
        self.train = train
        if train:
            self.N, self.vocab_size = self.train_classfication.shape
        else:
            self.N, self.vocab_size = self.test_classfication.shape
        self.label = data['label']

    def __getitem__(self, index):
        if self.train:
            return torch.from_numpy(np.squeeze(self.train_classfication[index])).float(),\
               torch.tensor(self.label[index])
        else:
            return torch.from_numpy(np.squeeze(self.test_classfication[index])).float(),\
               torch.tensor(self.label[self.spilit_sapce+index])

    def __len__(self):
        return self.N