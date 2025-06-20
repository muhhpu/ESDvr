import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from scipy.ndimage import gaussian_filter1d
from .mvgae import GCN

class TRY(GeneralRecommender):
    def __init__(self, config, dataset, logger=None):
        super(TRY, self).__init__(config, dataset)
        self.logger = logger
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.L_intereset = config['L_intereset']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        #     高斯
        num_user = self.n_users
        num_item = self.n_items
        num_layer = self.n_layers

        self.sigma1 = 0.4
        self.sigma2 = 0.8
        self.concate_way = config['mixture_type']
        self.dif_gats = config['num_dif_gats']
        if self.concate_way == 'add&dif':  # concate((title+poster),(poster-title))
            self.GD_gate = nn.Linear(self.embedding_dim * 2, self.dif_gats + 1)
        elif self.concate_way == 'concate&dif':  # concate(concate(title,poster),(poster-title))
            self.GD_gate = nn.Linear(self.embedding_dim * 3, self.dif_gats + 1)
        elif self.concate_way == 'dif':  # (poster-title)
            self.GD_gate = nn.Linear(self.embedding_dim, self.dif_gats + 1)
        self.GD_topk = config['GD_topk']
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.batch_size = config['train_batch_size']
        self.dim_x = config['embedding_size']
        self.beta = config['beta']
        self.aggr_mode = 'mean'
        self.concate = False
        if self.v_feat is not None:
            self.v_gcn = GCN(self.device, self.v_feat, self.edge_index, self.batch_size, num_user, num_item, self.dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.device, self.t_feat, self.edge_index, self.batch_size, num_user, num_item, self.dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)
        if self.video_feat is not None:
            self.video_gcn = GCN(self.device, self.video_feat, self.edge_index, self.batch_size, num_user, num_item,
                                 self.dim_x,
                                 self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)
        if self.frame_feat is not None:
            self.frame_gcn = GCN(self.device, self.frame_feat, self.edge_index, self.batch_size, num_user, num_item,
                                 self.dim_x,
                                 self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)



        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.k = 40
        self.tau = 0.5
        self.save = False
        self.epoch = 0
        self.frame_gate = config['frame_gate_type']
        self.wrong = config['wrong']
        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_x)
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                       allow_pickle=True).item()
        self.userOritem = config['userOritem']
        self.user_aggr_mode = 'softmax'
        self.additive_attention = AdditiveAttention(64,64)
        self.no_video_inter = False  # 视频不带交互信息
    #     frame gate
        self.gate_threshold = nn.Parameter(torch.full((num_item, 1), 100.0, dtype=torch.float32, requires_grad=True).to(self.device))
        self.gate_f = nn.Sequential(
            nn.Linear(4*self.embedding_dim, 1),
            # nn.Sigmoid()
        )
        self.user_pre_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_pre_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_pre_embedding.weight)
        nn.init.xavier_uniform_(self.item_pre_embedding.weight)

        self.storylamda =  nn.Parameter(torch.tensor(1.0))

    def pre_epoch_processing(self):
        if self.userOritem == 'user':
            self.epoch_item_graph, self.item_weight_matrix = self.topk_sample(self.k,self.user_graph_dict)
            self.item_weight_matrix = self.item_weight_matrix.to(self.device)
        else:
            self.epoch_item_graph, self.item_weight_matrix = self.topk_sample(self.k, self.item_graph_dict)
            self.item_weight_matrix = self.item_weight_matrix.to(self.device)
        pass
    def topk_sample(self, k,user_graph_dict):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)

        for i in range(len(user_graph_dict)):
            # length = len(user_graph_dict[i][1])
            # user_graph_dict[i][1] = [x / length for x in user_graph_dict[i][1]]
            if len(user_graph_dict[i][0]) < k:
                count_num += 1 # 1200
                if len(user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = user_graph_dict[i][0][:k]
                user_graph_weight = user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    # pdb.set_trace()
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                # user_weight_matrix[i] = torch.tensor(user_graph_weight) / sum(user_graph_weight) #weighted
                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                # pdb.set_trace()
                continue
            user_graph_sample = user_graph_dict[i][0][:k]
            user_graph_weight = user_graph_dict[i][1][:k]

            # user_weight_matrix[i] = torch.tensor(user_graph_weight) / sum(user_graph_weight) #weighted
            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                # pdb.set_trace()
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            # user_weight_list.append(user_weight)
            user_graph_index.append(user_graph_sample)
        # pdb.set_trace()
        return user_graph_index, user_weight_matrix
    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))
    def get_adj_mat(self):
        # 2.3
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats, _ = self.v_gcn()
        if self.t_feat is not None:
            text_feats, _ = self.t_gcn()


        if not self.wrong:
            image_feats = image_feats[self.n_users:]
            text_feats = text_feats[self.n_users:]
        else:
            image_feats = image_feats[:self.n_items]
            text_feats = text_feats[:self.n_items]



        if self.frame_gate != 'no':
            story , _ = self.video_gcn()


            if not self.wrong:
                story = story[self.n_users:]
            else:
                story = story[:self.n_items]



        smoothed_image_data1 = gaussian_filter1d(image_feats.cpu().detach().numpy(), sigma=self.sigma1)
        smoothed_text_data1 = gaussian_filter1d(text_feats.cpu().detach().numpy(), sigma=self.sigma1)
        smoothed_image_data2 = gaussian_filter1d(image_feats.cpu().detach().numpy(), sigma=self.sigma2)
        smoothed_text_data2 = gaussian_filter1d(text_feats.cpu().detach().numpy(), sigma=self.sigma2)

        dif1 = smoothed_image_data1 - smoothed_text_data1
        dif2 = smoothed_image_data2 - smoothed_text_data2
        dif3 = smoothed_image_data1 - smoothed_text_data2
        dif4 = smoothed_image_data2 - smoothed_text_data1
        dif5 = smoothed_text_data1 - smoothed_image_data1
        dif6 = smoothed_text_data2 - smoothed_image_data2
        dif7 = smoothed_text_data2 - smoothed_image_data1
        dif8 = smoothed_text_data1 - smoothed_image_data2
        empty_var = np.zeros_like(dif1)
        if self.dif_gats == 8:
            differences = np.stack([dif1, dif2, dif3, dif4, dif5, dif6, dif7, dif8,empty_var], axis=1)
        elif self.dif_gats == 4:
            differences = np.stack([dif1, dif2, dif3, dif4, empty_var], axis=1)
        differences_tensor = torch.tensor(differences).to(self.device)

        if self.concate_way == 'single':
            dif1 = torch.tensor(dif1).to(self.device)
            dif = torch.cat([torch.zeros((self.n_users, self.embedding_dim)).to(self.device), dif1], dim=0)
        else:
            if self.concate_way == 'add&dif':
                gate_featrues = torch.cat([image_feats + text_feats, image_feats - text_feats], dim=-1)
            elif self.concate_way == 'concate&dif':
                gate_featrues = torch.cat([image_feats, text_feats, image_feats - text_feats], dim=-1)
            elif self.concate_way == 'dif':
                gate_featrues = torch.cat([image_feats - text_feats], dim=-1)


            gate_scores = self.GD_gate(gate_featrues)


            # noise_level = 1e-10
            # noise_np = np.random.uniform(low=-noise_level, high=noise_level, size=gate_scores.shape)
            # noise = torch.tensor(noise_np, dtype=gate_scores.dtype, device=gate_scores.device)
            # gate_scores = gate_scores + noise
            # topk_weight, topk_idx = torch.topk(gate_scores, k=self.GD_topk, dim=-1, sorted=False)


            # 原不加扰动
            topk_weight, topk_idx = torch.topk(gate_scores, k=self.GD_topk, dim=-1, sorted=False)

            if self.GD_topk == 1:
                expanded_idx = topk_idx.unsqueeze(-1).expand(-1, -1, differences_tensor.size(-1)).to(self.device)
                selected_elements = torch.gather(differences_tensor, 1, expanded_idx).squeeze(1).to(self.device)
            else:
                weights = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
                expanded_idx = topk_idx.unsqueeze(-1).expand(-1, -1, differences_tensor.size(-1))
                selected_elements = torch.gather(differences_tensor, 1, expanded_idx)
                selected_elements = (selected_elements * weights.unsqueeze(-1)).sum(dim=1)
            padding = torch.zeros((self.n_users, self.embedding_dim)).to(self.device)
            dif = torch.cat([padding, selected_elements], dim=0)

        # element-wise 相乘，获得物品id指导下的模态信息（默认不同模态间偏好一致，但是不同模态间表示是否相同）
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))

        # User-Item View 用两个随机初始化矩阵 通过 交互矩阵 迭代 生成一个（基于交互的）内容矩阵表达
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            # 稀疏矩阵adj和稠密矩阵相乘= step+1的表达
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # 2 step 平均加权获得经过graph增强后的embedding，高阶聚合邻居信息
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        image_item_embeds = self.user_graph(image_item_embeds, _, self.epoch_item_graph, self.item_weight_matrix)
        text_item_embeds = self.user_graph(text_item_embeds, _, self.epoch_item_graph, self.item_weight_matrix)


        # user embdding 直接是交互矩阵*物品模态阵
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Behavior-Aware Fuser （并没有基于用户个性化权重）
        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)

        # frame gate
        if self.frame_gate in ['fixed','learnable','gru','user_learnable']:
            if self.frame_gate == 'gru':
                concatedd = torch.cat(
                    [sep_image_embeds[self.n_users:].unsqueeze(1), sep_text_embeds[self.n_users:].unsqueeze(1),
                     common_embeds[self.n_users:].unsqueeze(1)], dim=1)
                concatedd = self.gru(concatedd)
                concatedd = torch.cat([concatedd.squeeze(), self.gate_threshold], dim=-1)
                after = (1 - DiffSoftmax(concatedd, hard=True)[:, 3:4]) * story
                gated_story = torch.cat([torch.zeros((self.n_users, self.embedding_dim)).to(self.device), after], dim=0)
            elif self.frame_gate == 'user_learnable':
                frame_gating = self.gate_f(
                    torch.cat(
                        [sep_image_embeds[self.n_users:], sep_text_embeds[self.n_users:], common_embeds[self.n_users:],self.item_pre_embedding.weight],
                        dim=-1))

                # 使用 DiffSoftmax 选择故事，调整 logits 的维度为 [n_user, 2]
                logits = torch.cat([self.gate_threshold_1, frame_gating], dim=-1)  # [n_user, 2]
                gating_weights = DiffSoftmax(logits, hard=True, dim=-1)[:, 1:2]  # 取第二列的权重 [n_user, 1]

                # 根据 gating 权重选择故事
                after_select_story = gating_weights.unsqueeze(1) * story.sum(dim=1,
                                                                             keepdim=True)  # [n_user, n_item, embedding_dim]
                gated_story = after_select_story
                # gated_story = torch.cat(
                #     [torch.zeros((self.n_users, self.embedding_dim)).to(self.device), after_select_story], dim=0)
            else:
                frame_gating = self.gate_f(
                    torch.cat(
                        [sep_image_embeds[self.n_users:], sep_text_embeds[self.n_users:], common_embeds[self.n_users:],self.item_pre_embedding.weight],
                        dim=-1))
                if self.frame_gate == 'fixed':
                    after_select_story = DiffSoftmax(
                        torch.concat([torch.full(frame_gating.shape, 200.0).to(self.device), frame_gating], 1),
                        hard=True)[:, 1:2] * story
                else:
                    after_select_story = DiffSoftmax(torch.concat([self.gate_threshold, frame_gating], 1), hard=True)[:,
                                         1:2] * story

                user_embeds = self.user_embedding.weight
                all_train = torch.cat(
                    [user_embeds, after_select_story], dim=0)
                user_his_video = torch.sparse.mm(adj, all_train)
                gated_story = torch.cat(
                    [user_his_video[:self.n_users], after_select_story], dim=0)

            side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds + dif + gated_story) / 4
        else:
            side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds + dif ) / 4

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        u=self.user_pre_embedding(torch.tensor(users).to(dtype=torch.long, device=self.user_pre_embedding.weight.device))
        p=self.item_pre_embedding(torch.tensor(pos_items).to(dtype=torch.long, device=self.user_pre_embedding.weight.device))
        n=self.item_pre_embedding(torch.tensor(neg_items).to(dtype=torch.long, device=self.user_pre_embedding.weight.device))
        loss1, _, _ = self.bpr_loss(u,p,n)


        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)
        if self.no_video_inter:
            cl_loss = self.InfoNCE(side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + self.L_intereset * loss1

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode,dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features1,feattures2,user_graph,user_matrix):
        index = user_graph
        u_features = features1[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre
def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AdditiveAttention(nn.Module):
    def __init__(self, query_dimention, candidate_dimention, writer=None, tag=None, names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_dimention, query_dimention)
        self.attention_query_vector = nn.Parameter(torch.empty(query_dimention).uniform_(-0.1, 0.1))
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, candidate_vector):
        # ***L*q_d
        temp = torch.tanh(self.linear(candidate_vector))
        # ***L 得到每个的权重
        # print("temp:",temp)
        candidate_weights = self.softmax(torch.matmul(temp, self.attention_query_vector))
        # print("candidate_weights:",candidate_weights)

        target = torch.bmm(candidate_weights.unsqueeze(dim=1), candidate_vector).squeeze(dim=1)
        # print("target:",target)
        return target