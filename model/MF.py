import numpy as np
import scipy.sparse as sp

import torch
from torch import nn
import torch.nn.functional as F
from .base import BasicModel
from ipdb import set_trace

def log_map(base, target):
        cos_theta = (base * target).sum(dim=-1, keepdim=True)
        cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_theta)
        
        proj = target - cos_theta * base
        sin_theta = torch.sqrt(1 - cos_theta**2)

        factor = torch.where(
            sin_theta < 1e-6,
            torch.ones_like(sin_theta), 
            theta / sin_theta          
        )

        v = factor * proj
        return v


class MF(BasicModel):
    def __init__(self, args, dataset):
        super().__init__()
        self.dataset = dataset
        self.__init_weight(args)

    def __init_weight(self, args):
        self.num_users  = self.dataset.user_num
        self.num_items  = self.dataset.item_num
        self.latent_dim = args.latent_dim
        self.weight_decay = args.weight_decay
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.MSEloss = nn.MSELoss()



    def loss(self, users, pos, neg, batch_var_pos=None):
        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos)
        neg_emb = self.embedding_item(neg)
        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        align_loss = self.alignment_geometric_uncertainty(users_emb, pos_emb, batch_var_pos)

        all_items = torch.cat([pos_emb, neg_emb], dim=0)
        item_uniform_loss = self.uniformity(all_items, 2)
        user_uniform_loss = self.uniformity(users_emb, 2)
        loss = align_loss + 0.01 * (item_uniform_loss + user_uniform_loss) / 2

        return loss
        # return loss

    def uniformity(self, x, t=2):
        sim_matrix = torch.mm(x, x.t())
        sim_matrix = torch.clamp(sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7)
        sq_dist = 2.0 * (1.0 - sim_matrix)
        potential = torch.exp(-t * sq_dist)
        n = x.size(0)
        mask_diag = torch.eye(n, device=x.device).bool()
        potential = potential.masked_fill(mask_diag, 0.0)
        loss = potential.sum() / (n * (n - 1))
        return loss.log()

    
    def getUsersRating(self, users):
        users_emb = self.embedding_user.weight[users]
        items_emb = self.embedding_item.weight
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    

    
    def alignment_geometric_uncertainty(self, user_e, item_e, variance):
        diff = user_e - item_e
        dist_sq = diff.norm(p=2, dim=-1).pow(2)
        
        if variance is None:
            return dist_sq.mean()
        sigma_sq = variance.squeeze() 
        epsilon = 1e-6
        scale_factor = 0.5
        weight = 1.0 / (scale_factor * sigma_sq + epsilon)
        weighted_loss = dist_sq * weight.detach()
        
        return weighted_loss.mean()
    
    

            





