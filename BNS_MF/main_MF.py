# Implement BNS for Matrix Factorization
# @author Bin Liu, Bang Wang 

import random
import numpy as np
import pandas as pd
import evaluation
import sampling as ns

project_data= r'.\100k.csv'
train_data = r'.\100k_train.csv'
test_data = r'.\100k_test.csv'

seed=0
random.seed(seed)
np.random.seed(seed)


def load_data(path):
    project_data = pd.read_csv(path,
                               header=0,
                               dtype='str',
                               sep=',')
    u_count = project_data['user'].nunique()
    i_count = project_data['item'].nunique()
    u_list = [str(i) for i in range(u_count)]
    i_list = [str(i) for i in range(i_count)]

    return u_count,i_count,u_list,i_list

u_count,i_count,u_list,i_list = load_data(project_data)

def load_train(path):
    train_data = pd.read_csv(path,
                             header=0,
                             dtype='str',
                             sep=',')
    train_dict = {}
    datapair =[]
    popularity = np.zeros(i_count)

    for i in train_data.itertuples():
        u, i, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        popularity [int(i)] += 1
        train_dict.setdefault(u,{})
        train_dict[u][i] = rating
        datapair.append((u,i))
    random.shuffle(datapair)
    prior = popularity / (sum(popularity))

    dict_negative_items = {}
    dict_negative_index = {}
    for u in u_list:
        try:
            positive_items = set(train_dict[u].keys())
            negative_items = set(i_list) - positive_items
            dict_negative_items[u] = list(negative_items)
            dict_negative_index[u] = [i in negative_items for i in i_list]
        except KeyError:
            pass
    return train_dict,prior,popularity,datapair,dict_negative_items, dict_negative_index

train_dict,prior,popularity,datapair,negative_items_dict, negative_index_dict= load_train(train_data)

# load test data as np.array
def load_test(path):
    test_data = pd.read_csv(path,
                            header=0,
                            dtype='str',
                            sep=',')
    label = np.zeros((u_count, i_count))
    for i in test_data.itertuples():
        u = getattr(i, 'user')
        i = getattr(i, 'item')
        label[int(u), int(i)] = 1
    return label
label =  load_test(test_data)

def sigmoid(x):
    if x > 0 :
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x) /(1+np.exp(x))

#training 
class BPRMF:
    lr = 0.01
    reg = 0.01
    d = 32

    miu = 0
    sigma = reg ** 0.5

    U = np.random.normal(loc=miu,
                         scale=sigma,
                         size=[u_count, d])
    V = np.random.normal(loc=miu,
                         scale=sigma,
                         size=[i_count, d])
    train_count = 100

    def train(self, data_pair):
        for epoch in range(1, self.train_count + 1):
            print('MF:',epoch,'epoch-training')
            for ui in data_pair:
                u = ui[0]
                i = ui[1]
                negative_items = negative_items_dict[u]
                negative_index = negative_index_dict[u]
                rating_vector = np.array(np.mat(self.U[int(u)]) * np.mat(self.V.T))[0]
                ################### STARTING NEGATIVE SAMPLING ################### 
                size = 5
                alpha = 5
                j = ns.bns(i,negative_items, negative_index, rating_vector,prior, size, alpha)
                r_uij = rating_vector[int(i)] - rating_vector[int(j)]
                # update U and V
                loss_func = 1 - sigmoid(r_uij)
                # update U and V
                self.U[int(u)] += self.lr * (loss_func * (self.V[int(i)] - self.V[int(j)]) - self.reg * self.U[int(u)])
                self.V[int(i)] += self.lr * (loss_func * self.U[int(u)] - self.reg * self.V[int(i)])
                self.V[int(j)] += self.lr * (loss_func * (-self.U[int(u)]) - self.reg * self.V[int(j)])
            score = evaluation.erase(np.array(np.mat(self.U) * np.mat(self.V.T)),train_dict)
            print(evaluation.topk(score, label, 5))
            print(evaluation.topk(score, label, 10))
            print(evaluation.topk(score, label, 20))
        return  evaluation.erase(np.mat(self.U) * np.mat(self.V.T),train_dict),np.mat(self.U),np.mat(self.V)
# rating matirx and embeddings
score,U,V= BPRMF().train(datapair)

