import random
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch


class Data(object):
    def __init__(self, path, num_user, num_item, status='train'):
        self.num_user = num_user
        self.num_item = num_item
        self.status = status
        if self.status == 'train':
            self.train_dict, self.prior, self.popularity, self.train_pair = self.load_train_data(path)
            self.train_pair = np.asarray(self.train_pair)
            self.train_user = [pair[0] for pair in self.train_pair]
            self.train_item = [pair[1] for pair in self.train_pair]
            self.length = len(self.train_pair)
           
            self.UserItemNet = csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                          shape=(self.num_user, self.num_item))
            self.Lap_mat, self.Adj_mat = self.build_graph()
        else:
            self.test_dict, self.test_label = self.load_test_data(path)

    def load_train_data(self, path):
        data = pd.read_csv(path, header=0, sep=',')
        data_dict = {}
        datapair = []
        popularity = np.zeros(self.num_item)

        for i in data.itertuples():
            user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
            user, item = int(user), int(item)
            popularity[int(item)] += 1
            data_dict.setdefault(user, {})
            data_dict[user][item] = 1
            datapair.append((user, item))
        prior = popularity / sum(popularity)
        random.shuffle(datapair)

        return data_dict, prior, popularity**0.75, datapair

    def load_test_data(self, path):
        data = pd.read_csv(path, header=0, sep=',')
        label = np.zeros((self.num_user, self.num_item))
        data_dict = {}
        for i in data.itertuples():
            user, item = getattr(i, 'user'), getattr(i, 'item')
            data_dict.setdefault(user, set())
            data_dict[user].add(item)
            label[user, item] = 1
        return data_dict, label

    def build_graph(self):
        print('building graph adjacency matrix')
        st = time.time()
        adj_mat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.num_user, self.num_user:] = R
        adj_mat[self.num_user:, :self.num_user] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))

        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time.time()
        print(f"costing {end - st}s, obtained norm_mat...")

        return norm_adj, adj_mat

    def generate_batch(self, batch_size):
        n_batch = self.length // batch_size
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slices(self, index):
        pairs = self.train_pair[index]
        users, items = [], []
        for u, i in pairs:
            users.append(u)
            items.append(i)
        return users, items


def get_number_of_users_items(file):
    data = pd.read_csv(file, header=0, dtype='str', sep=',')
    userlist = list(data['user'].unique())
    itemlist = list(data['item'].unique())
    num_users, num_items = len(userlist), len(itemlist)

    return num_users, num_items


def convert_spmat_to_sptensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))




def get_uninteracted_item(train_dict, num_users, num_items):
    all_items = list(range(num_items))
    uninteracted_dict = {}
    num_uninter = []
    for user in range(num_users):
        interacted_items = set(train_dict[user].keys())
        uninteracted_items = set(all_items) - interacted_items
        uninteracted_dict[user] = list(uninteracted_items)
        num_uninter.append(len(uninteracted_items))
    return uninteracted_dict, num_uninter
