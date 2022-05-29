
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class LightGCNWithNG(nn.Module):
    def __init__(self, num_users,
                 num_items,
                 g_laplace,
                 g_adj,
                 prior,
                 popularity,
                 uninter_mat,
                 num_uninter,
                 opt,
                 device='cpu'):
        super(LightGCNWithNG, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.g_laplace = g_laplace
        self.g_adj = g_adj
        self.device = device

        self.dim = opt.dim
        self.hop = opt.hop
        self.num_negatives = opt.num_negatives
        self.alpha = opt.alpha
        self.prior = prior                          # (|V|,)
        self.popularity = popularity
        self.uninter_mat = uninter_mat
        self.num_uninter = num_uninter

        self.User_Emb = nn.Embedding(self.num_users, self.dim)
        nn.init.xavier_normal_(self.User_Emb.weight)
        self.Item_Emb = nn.Embedding(self.num_items, self.dim)
        nn.init.xavier_normal_(self.Item_Emb.weight)

        # LightGCN Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = LightGCNAgg(self.dim)
            self.add_module('Agg_LightGCN_{}'.format(i), agg)
            self.global_agg.append(agg)

    def computer(self):
        users_emb = self.User_Emb.weight
        items_emb = self.Item_Emb.weight
        all_emb = torch.cat((users_emb, items_emb), dim=0)
        embs = [all_emb]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(A=self.g_laplace, x=embs[i])
            embs.append(x)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))


    def proposed_negative_sample(self, users,items, ui_scores):
        batch_size = users.size(0)
        if self.device == 'cpu':
            users = users.detach().numpy()
            ui_scores = ui_scores.detach().numpy()
        else:
            users = users.cpu().detach().numpy()
            ui_scores = ui_scores.cpu().detach().numpy()
        negatives = []
        for bs in range(batch_size):
            u = users[bs]
            i = items[bs]
            rating_vector = ui_scores[bs]
            x_ui = rating_vector[i]
            negative_items = self.uninter_mat[u]

            candidate_set = np.random.choice(negative_items, size=self.num_negatives, replace=False)
            candidate_scores = [rating_vector[l] for l in candidate_set]

            # step 1 : computing info(l)
            info = np.array([1 - self.sigmoid(x_ui - x_ul) for x_ul in candidate_scores])  # O(1)
            # step 2 : computing prior probability
            p_fn = np.array([self.prior[l] for l in candidate_set])  # O(1)
            # step 3 : computing empirical distribution function (likelihood)
            F_n = np.array([np.sum(rating_vector <= x_ul) / (self.num_items+1) for x_ul in candidate_scores])  # O(|I|)
            # step 4: computing posterior probability
            unbias = (1 - F_n) * (1 - p_fn) / (1 - F_n - p_fn + 2 * F_n * p_fn)  # O(1)
            # step 5: computing conditional sampling risk
            conditional_risk = (1 - unbias) * info - self.alpha * unbias * info  # O(1)
            j = candidate_set[conditional_risk.argsort()[0]]
            negatives.append(j)
        negatives = torch.LongTensor(negatives)
        negatives = negatives.to(self.device)
        return negatives

    def forward(self, epoch,users, items):
        all_users_emb, all_items_emb = self.computer()      # |U| * d, |V| * d
        users_emb = all_users_emb[users]    # bs * d
        items_emb = all_items_emb[items]    # bs * d

        ui_scores = torch.mm(users_emb, all_items_emb.t())  # bs * |V|
        negatives = self.proposed_negative_sample(users,items, ui_scores)  # bs
        neg_item_emb = all_items_emb[negatives]  # bs * d

        pos_scores = torch.mul(users_emb, items_emb)
        pos_scores = pos_scores.sum(dim=1)      # (bs,)
        neg_scores = torch.mul(users_emb, neg_item_emb)
        neg_scores = neg_scores.sum(dim=1)      # (bs,)

        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return bpr_loss

    def predict(self):
        all_users_emb, all_items_emb = self.computer()      # |U| * d, |V| * d
        rate_mat = torch.mm(all_users_emb, all_items_emb.t())
        return rate_mat


class LightGCNAgg(nn.Module):
    def __init__(self, hidden_size):
        super(LightGCNAgg, self).__init__()
        self.dim = hidden_size

    def forward(self, A, x):
        '''
            A: n \times n
            x: n \times d
        '''
        return torch.sparse.mm(A, x)



