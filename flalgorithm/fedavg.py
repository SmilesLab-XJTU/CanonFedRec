import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
from utils.logger import *
from utils.utils import *
from utils.metric import *
from ipdb import set_trace
from .base import Base
from .smiles import *


class Fedavg(Base):
    def build_model(self, args):
        early_stop_num = args.early_stop_num
        lr = args.lr

        self.client_model = list()
        self.client_optimizer = list()
        for i in range(self.client_num):
            self.client_model.append(get_model(args, self.dataset_list[i]))
            self.client_optimizer.append(torch.optim.Adam(params=self.client_model[i].parameters(), lr=lr))
        self.server_model = get_model(args, self.dataset_list[0])
        self.server_paramname = [k for k in self.server_model.state_dict() if 'user' not in k]
        self.early_stopper = EarlyStopper_base(num_trials=early_stop_num, save_path=self.exp_save_name, client_num=self.client_num)
        self.global_item_variance = None

    def build_global_config(self, args):
        self.epoch = args.epoch
        self.update_frequence = args.update_frequency

    def client_init(self):
        # set_trace()
        for i in range(self.client_num):
            private_state = {k:self.server_model.state_dict()[k] for k in self.server_paramname}
            self.client_model[i].load_state_dict(private_state, strict=False)

    def train_client_model(self):
        global_var = copy.deepcopy(self.global_item_variance)
        batch_var_pos = None
        batch_var_neg = None
        for i in range(self.client_num):
            self.client_model[i] = self.client_model[i].to(self.device)
            self.client_model[i].train()
            for client_epoch in range(self.update_frequence):#local epoch=1
                for (batch_users, batch_pos, batch_neg) in self.loader_list[i]:
                    batch_users, batch_pos, batch_neg = batch_users.to(self.device), batch_pos.to(self.device), batch_neg.to(self.device)
                    
                    if global_var is not None:
                        batch_var_pos = self.global_item_variance[batch_pos]
                        batch_var_neg = self.global_item_variance[batch_neg]


                    loss = self.client_model[i].loss(batch_users, batch_pos, batch_neg, batch_var_pos)
                    self.client_optimizer[i].zero_grad()
                    loss.backward()
                    self.client_optimizer[i].step()
            self.client_model[i] = self.client_model[i].to('cpu')

    
    def client_upload(self):
        datasize_sum = 0
        aggstate = {k:0.0 for k in self.server_paramname}

        for i in range(self.client_num):
            datasize = self.dataset_list[i].trainnum
            datasize_sum += datasize
            self.client_model[i] = self.client_model[i].to(self.device)
            for k in self.server_paramname:
                aggstate[k] = aggstate[k] + self.client_model[i].state_dict()[k] * datasize
            self.client_model[i] = self.client_model[i].to("cpu")

        for k in self.server_paramname:
            aggstate[k] = aggstate[k] / datasize_sum

        self.server_model.load_state_dict(aggstate, strict=False)

    def calculate_semantic(self, anchor_items, last_global_item):
        key_param = 'embedding_item.weight'
        tangent_vectors = []
        datasize_sum = 0
        anchor_items = anchor_items.to(self.device)
        anchor_items = F.normalize(anchor_items, p=2, dim=-1)
        v_agg = torch.zeros_like(anchor_items, device=anchor_items.device)
        data_weights = []
        active_tangent_sum = torch.zeros_like(anchor_items)
        active_tangent_sq_sum = torch.zeros_like(anchor_items) 
        active_weight_sum = torch.zeros(anchor_items.shape[0], device=self.device)
        for i in range(self.client_num):
            datasize = self.dataset_list[i].trainnum
            datasize_sum += datasize
            data_weights.append(datasize)
            self.client_model[i] = self.client_model[i].to(self.device)
            client_item = self.client_model[i].state_dict()[key_param].detach()

            active_ids = torch.argwhere((client_item - last_global_item).norm(2, dim=-1) > 1e-6).squeeze(1)
            client_item = active_orthogonal_procrustes(anchor_items, client_item, active_ids)
            client_item = F.normalize(client_item, p=2, dim=-1)
            
            log_map_client = log_map(anchor_items, client_item)
            
            v_agg += log_map_client * datasize
            if active_ids.numel() > 0:
                raw_active_vec = log_map_client[active_ids]
                weighted_vec = raw_active_vec * datasize
                active_tangent_sum.index_add_(0, active_ids, weighted_vec)
                weighted_sq_vec = (raw_active_vec ** 2) * datasize
                active_tangent_sq_sum.index_add_(0, active_ids, weighted_sq_vec)
                weight_update = torch.full((active_ids.shape[0],), float(datasize), device=self.device, dtype=torch.float)

                active_weight_sum.index_add_(0, active_ids, weight_update)

        gamma=3
        v_agg = v_agg / datasize_sum
        v_agg = v_agg * gamma
        mu_sem = exp_map(anchor_items, v_agg)

        valid_mask = active_weight_sum > 1e-6
        safe_weight_sum = active_weight_sum.clone()
        safe_weight_sum[~valid_mask] = 1.0
        safe_weight_sum = safe_weight_sum.unsqueeze(-1)
        mean_active = active_tangent_sum / safe_weight_sum
        sq_mean_active = active_tangent_sq_sum / safe_weight_sum
        variance_vec = sq_mean_active - mean_active ** 2
        variance_vec[~valid_mask] = 0.0
        variance_vec = variance_vec.sum(dim=1).clamp(min=1e-8)
        variance_vec = variance_vec * (gamma**2)
        self.global_item_variance = variance_vec.detach()

        del anchor_items
        return mu_sem

    
    def server_round(self):
        key_param = 'embedding_item.weight'
        datasize_sum = 0
        aggstate = {k:0.0 for k in self.server_paramname}
        global_item = self.server_model.state_dict()[key_param].to(self.device)
        last_global_item = copy.deepcopy(self.server_model.state_dict()[key_param]).to(self.device)

        global_semantic = self.calculate_semantic(global_item, last_global_item)
        fused_embedding =  global_semantic

        for i in range(self.client_num):
            datasize = self.dataset_list[i].trainnum
            datasize_sum += datasize
            self.client_model[i] = self.client_model[i].to(self.device)
            for k in self.server_paramname:
                if k != key_param:
                    aggstate[k] = aggstate[k] + self.client_model[i].state_dict()[k] * datasize
            self.client_model[i] = self.client_model[i].to("cpu")

        for k in self.server_paramname:
            aggstate[k] = aggstate[k] / datasize_sum
        
        aggstate[key_param] = fused_embedding
        self.server_model.load_state_dict(aggstate, strict=False)


    def client_download(self):
        for i in range(self.client_num):
            private_state = {k:self.server_model.state_dict()[k] for k in self.server_paramname}
            self.client_model[i].load_state_dict(private_state, strict=False)

    def model_save(self, step):
        for i in range(self.client_num): 
            torch.save(self.client_model[i].state_dict()['embedding_item.weight'], self.exp_save_name + '/clients/item_' + step + str(i) + '.pth')

    def fit(self):

        for epoch_i in range(self.epoch):
            start_time = time.time()
            
            # Train client model
            self.epoch_i = epoch_i
            
            # Init model
            if epoch_i == 0:
                self.client_init()
                self.logger.info(f"Finish init client model in {(time.time() - start_time):.2f}s")
            
            self.train_client_model()
            self.logger.info(f"Finish client train in {(time.time() - start_time):.2f}s")

            # Upload server from client
            self.server_round()

            # self.client_upload()
            self.logger.info(f"Finish upload in {(time.time() - start_time):.2f}s")

            # Download client from server
            self.client_download()
            self.logger.info(f"Finish download in {(time.time() - start_time):.2f}s")

            # Test server model and Early stop
            valid_metric = self.evaluate_valid()
            self.logger.info(f"Finish valid valid set epoch {epoch_i} in {(time.time() - start_time):.2f}s, f1: {valid_metric[2]}, mrr: {valid_metric[3]}, ndcg: {valid_metric[4]}.")
            if not self.early_stopper.is_continuable(self.client_model, valid_metric[2][0], all_accuracy=valid_metric):
                self.logger.info(f'early stop at epoch {epoch_i}')
                self.logger.info(f'validation: best f1: {self.early_stopper.best_accuracy}')
                self.client_model = self.early_stopper.model_load(self.client_model)
                break

        test_metric = self.evaluate_test()
        self.logger.info(f"f1: {test_metric[2]}, mrr: {test_metric[3]}, ndcg: {test_metric[4]}.")
        self.writer.close()
