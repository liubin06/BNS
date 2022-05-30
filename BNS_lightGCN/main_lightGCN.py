
import argparse
import time
import datetime
import numpy as np
import torch
import sys
import os
import math
import heapq
from data import *
from model import *
print(torch.__version__)


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='100k', help='dataset')
'''训练基本参数'''
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_epoch', type=list, default=[20, 40, 60, 80], help='the epoch which the learning rate decay')
parser.add_argument('--patience', type=int, default=5)

parser.add_argument('--dim', type=int, default=32, help='dimension of vector')
parser.add_argument('--hop', type=int, default=1, help='number of LightGCN layers')

##################
parser.add_argument('--num_negatives', type=int, default=5, help='number of negative instances')
parser.add_argument('--alpha', type=int, default=5, help='weight')
parser.add_argument('--topk', type=list, default=[5, 10, 20], help='length of recommendation list')
parser.add_argument('--log', type=bool, default=False)

opt = parser.parse_args()

if opt.log:
    path = 'log/' + opt.dataset
    if not os.path.exists(path):
        os.makedirs(path)
    file = path + '/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_hop' + str(opt.hop) + '.txt'
    f = open(file, 'w')
else:
    f = sys.stdout

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


def main():
    t0 = time.time()
    init_seed(2022)
    if opt.dataset == '100k':
        total_file = 'datasets/' + '/100k.csv'
        train_file = 'datasets/' +  '/100k_train.csv'
        test_file = 'datasets/'  +  '/100k_test.csv'


    num_users, num_items = get_number_of_users_items(total_file)
    train_data = Data(train_file, num_users, num_items, status='train')
    test_data = Data(test_file, num_users, num_items, status='test')

    train_slices = train_data.generate_batch(opt.batch_size)

    G_Lap_tensor = convert_spmat_to_sptensor(train_data.Lap_mat)
    G_Adj_tensor = convert_spmat_to_sptensor(train_data.Adj_mat)
    G_Lap_tensor = G_Lap_tensor.to(device)
    G_Adj_tensor = G_Adj_tensor.to(device)


    uninter_mat, num_uninter = get_uninteracted_item(train_data.train_dict, num_users, num_items)

    model = LightGCNWithNG(num_users, num_items, G_Lap_tensor, G_Adj_tensor, train_data.prior,train_data.popularity,
                           uninter_mat, num_uninter, opt, device)
    model = model.to(device)
    print(model, file=f)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_dc_epoch, gamma=opt.lr_dc)

    best_result = {}
    best_epoch = {}
    for k in opt.topk:
        best_result[k] = [0., 0., 0., 0.]
        best_epoch[k] = [0, 0, 0, 0]
    # bad_counter = 0
    for epoch in range(opt.epochs):
        st = time.time()
        print('-------------------------------------------', file=f)
        print('epoch: ', epoch, file=f)
        pre_dic, rec_dic, F1_dict, ndcg_dict = train_test(model, train_data, test_data, train_slices, optimizer,epoch)
        scheduler.step()
        for k in opt.topk:
            if pre_dic[k] > best_result[k][0]:
                best_result[k][0] = pre_dic[k]
                best_epoch[k][0] = epoch
            if rec_dic[k] > best_result[k][1]:
                best_result[k][1] = rec_dic[k]
                best_epoch[k][1] = epoch
            if F1_dict[k] > best_result[k][2]:
                best_result[k][2] = F1_dict[k]
                best_epoch[k][2] = epoch
            if ndcg_dict[k] > best_result[k][3]:
                best_result[k][3] = ndcg_dict[k]
                best_epoch[k][3] = epoch
            print('Pre@%d:\t%0.4f\tRecall@%d:\t%0.4f\tF1@%d:\t%0.4f\tNDCG@%d:\t%0.4f\t[%0.2f s]' %
                  (k, pre_dic[k], k, rec_dic[k], k, F1_dict[k], k, ndcg_dict[k], (time.time() - st)), file=f)

    print('------------------best result-------------------', file=f)
    for k in opt.topk:
        print('Best Result: Pre@%d: %0.4f\tRecall@%d: %0.4f\tF1@%d: \t%0.4f\tNDCG@%d: \t%0.4f [%0.2f s]' %
              (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3], (time.time() - t0)), file=f)
        print('Best Epoch: Pre@%d: %d\tRecall@%d: %d\tF1@%d: %d\tNDCG@%d: %d\t [%0.2f s]' % (
            k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3], (time.time() - t0)), file=f)
    print('------------------------------------------------', file=f)
    print('Run time: %0.2f s' % (time.time() - t0), file=f)


def train_test(model, train_data, test_data, train_slices, optimizer,epoch):
    print('start training: ', datetime.datetime.now(), file=f)
    model.train()
    total_loss = []
    for index in train_slices:
        optimizer.zero_grad()
        users, items = train_data.get_slices(index)
        users = torch.LongTensor(users).to(device)
        items = torch.LongTensor(items).to(device)

        bpr_loss = model(epoch,users, items)

        bpr_loss.backward()

        optimizer.step()

        total_loss.append(bpr_loss.item())

    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']), file=f)

    print('----------------', file=f)
    print('start predicting: ', datetime.datetime.now(), file=f)
    model.eval()
    pre_dic, rec_dic, F1_dict, ndcg_dict = {}, {}, {}, {}

    rating_mat = model.predict()    # |U| * |V|
    if device == 'cpu':
        rating_mat = rating_mat.detach().numpy()
    else:
        rating_mat = rating_mat.cpu().detach().numpy()
    rating_mat = erase(rating_mat, train_data.train_dict)

    for k in opt.topk:
        metrices = topk_eval(rating_mat, test_data.test_label, k)
        precision, recall, F1, ndcg = metrices[0], metrices[1], metrices[2], metrices[3]
        pre_dic[k] = precision
        rec_dic[k] = recall
        F1_dict[k] = F1
        ndcg_dict[k] = ndcg

    return pre_dic, rec_dic, F1_dict, ndcg_dict


def erase(score, train_dict):
    for user in train_dict:
        for item in train_dict[user]:
            score[user, item] = -1000.0
    return score


def topk_eval(score, label, k):
    '''
    :param score: prediction
    :param k: number of top-k
    '''
    evaluation = [0, 0, 0, 0]
    counter = 0
    discountlist = [1 / math.log(i + 1, 2) for i in range(1, k + 1)]

    for user_no in range(score.shape[0]):
        user_score = score[user_no].tolist()
        user_label = label[user_no].tolist()
        label_count = int(sum(user_label))
        topn_recommend_score = heapq.nlargest(k, user_score)  
        topn_recommend_index = [user_score.index(i) for i in
                                topn_recommend_score]  # map(user_score.index,topn_recommend_score)
        topn_recommend_label = [user_label[i] for i in topn_recommend_index]  
        idcg = discountlist[0:label_count]

        if label_count == 0:
            counter += 1
            continue
        else:
            topk_label = topn_recommend_label[0:k]
            true_positive = sum(topk_label)
            evaluation[0] += true_positive / k  # precision
            evaluation[1] += true_positive / label_count  # recall
            evaluation[2] += 2 * true_positive / (k + label_count)  # f1
            evaluation[3] += np.dot(topk_label, discountlist[0:]) / sum(idcg)  # ndcg
    return [i / (score.shape[0] - counter) for i in evaluation]


main()



