# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
acc_func = lambda x,y: np.sum(x == y) / len(x)


mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class MF(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
           
    def fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out      

        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()
    


class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        out = self.sigmoid(self.linear_1(z_emb))


        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)            
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class Embedding_Sharing(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)


        if is_training:
            return torch.squeeze(z_emb), U_emb, V_emb
        else:
            return torch.squeeze(z_emb)        
    
    
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size / 2, bias = False)
        self.linear_2 = torch.nn.Linear(input_size / 2, 1, bias = True)
        self.xent_func = torch.nn.BCELoss()        
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        
        return torch.squeeze(x)   






class MF_Cali_MR(nn.Module):
    """
    校准的多重鲁棒模型（Cali-MR）。
    """
    def __init__(self, num_users, num_items, embedding_k=4, J=2, K=2, M=10, *args, **kwargs):
        super(MF_Cali_MR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k

        self.J = J  # 倾向评分模型的数量
        self.K = K  # 插补模型的数量
        self.M = M  # 软分桶的数量

        # 预测模型
        self.prediction_model = MF_BaseModel(num_users, num_items, embedding_k)

        # 插补模型
        self.imputation_models = nn.ModuleList([
            MF_BaseModel(num_users, num_items, embedding_k)
            for _ in range(self.K)
        ])

        # 倾向评分模型
        self.propensity_models = nn.ModuleList([
            MF_BaseModel(num_users, num_items, embedding_k)
            for _ in range(self.J)
        ])

        # 软分桶层
        self.propensity_soft_binning = nn.Linear(in_features=1, out_features=self.M)
        self.imputation_soft_binning = nn.Linear(in_features=1, out_features=self.M)

        # 可学习的权重参数
        self.alpha = torch.nn.Parameter(torch.randn(self.J))  # 倾向评分模型的权重参数，形状为 (J,)
        self.beta = torch.nn.Parameter(torch.randn(self.K))   # 插补模型的权重参数，形状为 (K,)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()



    # batch_size_prop调大可提升实验速度
    def _compute_IPS(self, x, batch_size_prop=2048, num_epoch=100, lr=0.05, lamb=0, gamma=1, tol=1e-4, verbose=False):
        """
        计算逆倾向评分（Inverse Propensity Score，IPS）。
        """
        print("compute_IPS")
        # 构建观察矩阵
        obs = sps.csr_matrix(
            (np.ones(x.shape[0]), (x[:, 0], x[:, 1])),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        ).toarray().reshape(-1)  # (num_users * num_items,)

        # 定义优化器列表，用于优化倾向评分模型
        optimizer_propensity_list = [
            torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)
            for model in self.propensity_models
        ]
        # 软分桶层的优化器
        optimizer_propensity_soft_binning = torch.optim.Adam(
            self.propensity_soft_binning.parameters(), lr=lr, weight_decay=lamb
        )
        # 权重参数 alpha 的优化器
        optimizer_alpha = torch.optim.Adam([self.alpha], lr=lr, weight_decay=lamb)

        last_loss = 1e9
        num_sample = len(obs)
        total_batch = num_sample // batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)  # (num_users * num_items, 2)
        early_stop = 0

        for epoch in range(num_epoch):
            print("IPS")
            print(epoch)
            # Shuffle all samples
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # Get batch samples
                x_all_idx = ul_idxs[idx * batch_size_prop : (idx + 1) * batch_size_prop]
                x_sampled = x_all[x_all_idx]  # (batch_size_prop, 2)

                sub_obs = obs[x_all_idx]  # (batch_size_prop,)
                sub_obs = torch.Tensor(sub_obs).cuda()

                # Compute propensity scores for all propensity models
                prop_outputs = [
                    model.forward(x_sampled).squeeze()  # (batch_size_prop,)
                    for model in self.propensity_models
                ]
                prop_outputs = torch.stack(prop_outputs, dim=1)  # (batch_size_prop, J)

                # Weighted combination
                w_prop = F.softmax(self.alpha, dim=0)  # (J,)
                prop = torch.matmul(prop_outputs, w_prop)  # (batch_size_prop,)

                # Compute loss for weighted propensity scores
                loss_w = F.binary_cross_entropy(prop, sub_obs)

                # Compute gradient for alpha
                alpha_grad = torch.autograd.grad(loss_w, [self.alpha], create_graph=True)[0]  # (J,)
                updated_alpha = self.alpha - lr * alpha_grad  # (J,)

                # Compute propensity loss for each model
                prop_losses = [
                    F.binary_cross_entropy(prop_outputs[:, j], sub_obs, reduction="mean")
                    for j in range(self.J)
                ]
                total_prop_loss = sum(prop_losses)  # scalar

                # Compute updated weighted propensity scores
                updated_w_prop = F.softmax(updated_alpha, dim=0)  # (J,)
                updated_prop = torch.matmul(prop_outputs, updated_w_prop).view(-1, 1)  # (batch_size_prop, 1)

                # Soft binning
                prop_soft_binning = F.softmax(self.propensity_soft_binning(updated_prop), dim=-1)  # (batch_size_prop, M)

                # Compute calibration loss
                acc_m = torch.sum(prop_soft_binning * sub_obs.unsqueeze(1), dim=0)  # (M,)
                conf_m = torch.sum(prop_soft_binning * updated_prop, dim=0)         # (M,)
                prop_dece_m = torch.abs(acc_m - conf_m)                             # (M,)

                prop_dece_loss = torch.mean(prop_dece_m)  # scalar

                # Total propensity loss
                loss_prop = total_prop_loss + gamma * prop_dece_loss

                # Backpropagate and update propensity models
                for optimizer_propensity in optimizer_propensity_list:
                    optimizer_propensity.zero_grad()
                loss_prop.backward(retain_graph=True)
                for optimizer_propensity in optimizer_propensity_list:
                    optimizer_propensity.step()

                # Update alpha
                prop_outputs = [
                    model.forward(x_sampled).squeeze()
                    for model in self.propensity_models
                ]
                prop_outputs = torch.stack(prop_outputs, dim=1)  # (batch_size_prop, J)
                w_prop = F.softmax(self.alpha, dim=0)  # (J,)
                prop = torch.matmul(prop_outputs, w_prop)  # (batch_size_prop,)
                loss_w = F.binary_cross_entropy(prop, sub_obs)

                optimizer_alpha.zero_grad()
                loss_w.backward(retain_graph=True)
                optimizer_alpha.step()

                # Update soft binning layers
                updated_w_prop = F.softmax(self.alpha, dim=0)  # (J,)
                prop = torch.matmul(prop_outputs, updated_w_prop).view(-1, 1)  # (batch_size_prop, 1)

                prop_soft_binning = F.softmax(self.propensity_soft_binning(prop), dim=-1)  # (batch_size_prop, M)

                acc_m = torch.sum(prop_soft_binning * sub_obs.unsqueeze(1), dim=0)
                conf_m = torch.sum(prop_soft_binning * prop, dim=0)

                prop_dece_m = torch.abs(acc_m - conf_m)
                prop_dece_loss = torch.mean(prop_dece_m)

                optimizer_propensity_soft_binning.zero_grad()
                prop_dece_loss.backward(retain_graph=True)
                optimizer_propensity_soft_binning.step()

                epoch_loss += (total_prop_loss + prop_dece_loss).detach().cpu().numpy()

            # Early stopping
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[PS] Reach preset epochs, it seems does not converge.")  
                

    def fit(self, x, y, unlabel_x, stab=0, num_epoch=100, batch_size=128, lr1=0.05, lamb1=1e-4, lr2=0.05, lamb2=5e-5,
            lr3=0.05, lamb3=5e-5, gamma=8, prop_clip=0.05, tol=1e-4, G=3, verbose=False):
        """
        训练模型，包括预测模型、插补模型和校准机制。
        """
        print("fit")
        # 定义优化器
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1
        )
        optimizer_imputation_list = [
            torch.optim.Adam(model.parameters(), lr=lr2, weight_decay=lamb2)
            for model in self.imputation_models
        ]
        optimizer_imputation_soft_binning = torch.optim.Adam(
            self.imputation_soft_binning.parameters(), lr=lr3, weight_decay=lamb3
        )
        optimizer_beta = torch.optim.Adam([self.beta], lr=lr1, weight_decay=lamb1)

        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        # Initialize weights
        w_prop = F.softmax(self.alpha, dim=0)  # (J,)
        early_stop = 0

        for epoch in range(num_epoch):
            print("loop")
            print(epoch)
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # Shuffle unlabeled data indices
            ul_idxs = np.arange(unlabel_x.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # 小批量训练
                selected_idx = all_idx[batch_size * idx : (idx + 1) * batch_size]
                sub_x = x[selected_idx]                # (batch_size, 2)
                sub_y = y[selected_idx]                # (batch_size,)
                sub_y = torch.Tensor(sub_y).cuda()     # (batch_size,)

                # 计算倾向评分和逆倾向评分
                prop_outputs = [
                    model.forward(sub_x).squeeze()  # (batch_size,)
                    for model in self.propensity_models
                ]
                prop_outputs = torch.stack(prop_outputs, dim=1)  # (batch_size, J)

                w_prop = F.softmax(self.alpha, dim=0)  # (J,)
                prop = torch.matmul(prop_outputs, w_prop)  # (batch_size,)

                inv_prop = 1 / torch.clamp(prop, min=prop_clip, max=1)  # (batch_size,)

                # 计算插补模型的输出
                imp_outputs = [
                    model.forward(sub_x).squeeze()  # (batch_size,)
                    for model in self.imputation_models
                ]
                imp_outputs = torch.stack(imp_outputs, dim=1)  # (batch_size, K)

                v_imp = F.softmax(self.beta, dim=0)  # (K,)
                imp_y = torch.matmul(imp_outputs, v_imp)  # (batch_size,)

                # 预测模型的输出
                pred, _, _ = self.prediction_model.forward(sub_x, is_training=True)  # (batch_size,)

                # Compute loss
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")  # (batch_size,)
                e_hat_loss = F.binary_cross_entropy(imp_y, pred, reduction="none")  # (batch_size,)
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()  # scalar

                # Compute gradient for beta
                beta_grad = torch.autograd.grad(imp_loss, [self.beta], create_graph=True)[0]  # (K,)
                updated_beta = self.beta - optimizer_beta.param_groups[0]['lr'] * beta_grad

                # Compute imputation loss
                imp_losses = [
                    (((e_loss - F.binary_cross_entropy(model.forward(sub_x).squeeze(), pred, reduction="none")) ** 2) * inv_prop).mean()
                    for model in self.imputation_models
                ]
                total_imp_loss = sum(imp_losses)

                # Compute calibration loss
                imp_soft_binning = F.softmax(self.imputation_soft_binning(torch.matmul(imp_outputs, v_imp).view(-1, 1)), dim=-1)  # (batch_size, M)

                acc_m = torch.sum(imp_soft_binning * e_loss.unsqueeze(1), dim=0) / torch.sum(inv_prop)
                conf_m = torch.sum(imp_soft_binning * torch.matmul(imp_outputs, v_imp).view(-1, 1), dim=0)

                imp_dece_m = torch.abs(acc_m - conf_m)
                imp_dece_loss = torch.mean(imp_dece_m)

                # Total imputation loss
                loss_imp = total_imp_loss + gamma * imp_dece_loss

                # Backpropagate and update imputation models
                for optimizer_imputation in optimizer_imputation_list:
                    optimizer_imputation.zero_grad()
                loss_imp.backward(retain_graph=True)
                for optimizer_imputation in optimizer_imputation_list:
                    optimizer_imputation.step()

                # Update beta
                loss_v = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                optimizer_beta.zero_grad()
                loss_v.backward(retain_graph=True)
                optimizer_beta.step()

                # Update soft binning layers
                v_imp = F.softmax(self.beta, dim=0)  # (K,)
                imp_y = torch.matmul(imp_outputs, v_imp).view(-1, 1)  # (batch_size, 1)

                imp_soft_binning = F.softmax(self.imputation_soft_binning(imp_y), dim=-1)  # (batch_size, M)

                acc_m = torch.sum(imp_soft_binning * e_loss.unsqueeze(1), dim=0) / torch.sum(inv_prop)
                conf_m = torch.sum(imp_soft_binning * imp_y, dim=0)

                imp_dece_m = torch.abs(acc_m - conf_m)
                imp_dece_loss = torch.mean(imp_dece_m)

                optimizer_imputation_soft_binning.zero_grad()
                imp_dece_loss.backward(retain_graph=True)
                optimizer_imputation_soft_binning.step()

                # 更新预测模型
                # Sample unlabeled data
                sampled_indices = G * idx * batch_size
                sampled_end = G * (idx + 1) * batch_size
                if sampled_end > unlabel_x.shape[0]:
                    sampled_end = unlabel_x.shape[0]
                x_sampled = unlabel_x[ul_idxs[sampled_indices : sampled_end]]  # (G*batch_size, 2)

                # Compute inverse propensity scores for sampled data
                inv_prop_all_list = [
                    1 / torch.clamp(model.forward(x_sampled), min=prop_clip, max=1)
                    for model in self.propensity_models
                ]
                inv_prop_all_tensor = torch.stack(inv_prop_all_list, dim=1)  # (num_sample_unlabeled, J)

                # Compute imputation predictions for sampled data
                imp_predictions_all = [
                    model.forward(x_sampled).squeeze()  # (num_sample_unlabeled,)
                    for model in self.imputation_models
                ]
                imp_predictions_all_tensor = torch.stack(imp_predictions_all, dim=1)  # (num_sample_unlabeled, K)

                # Construct matrix u_all for sampled data
                u_all = torch.cat([inv_prop_all_tensor, imp_predictions_all_tensor], dim=1).cuda()  # (num_sample_unlabeled, J + K)

                # Construct matrix u for observed data
                inv_prop_list = [
                    1 / torch.clamp(model.forward(sub_x), min=prop_clip, max=1)  # (batch_size,)
                    for model in self.propensity_models
                ]
                inv_prop_tensor = torch.stack(inv_prop_list, dim=1)  # (batch_size, J)

                imp_predictions = [
                    model.forward(sub_x).squeeze()  # (batch_size,)
                    for model in self.imputation_models
                ]
                imp_predictions_tensor = torch.stack(imp_predictions, dim=1)  # (batch_size, K)

                # Construct matrix u for observed data
                u = torch.cat([inv_prop_tensor, imp_predictions_tensor], dim=1).cuda()  # (batch_size, J + K)

                # Compute matrix inverse
                try:
                    matrix = u.T.matmul(u) + stab * torch.eye(self.J + self.K).cuda()
                    matrix_inv = torch.inverse(matrix)
                except RuntimeError as e:
                    print("Matrix inversion error:", e)
                    matrix = u.T.matmul(u) + (stab + 1e-6) * torch.eye(self.J + self.K).cuda()
                    matrix_inv = torch.inverse(matrix)

                # Compute eta
                e_loss_tensor = e_loss.view(-1, 1).cuda()  # (batch_size, 1)
                eta = torch.squeeze(matrix_inv.matmul(u.T.matmul(e_loss_tensor)))  # (J + K,)

                # Compute loss for prediction model
                loss = torch.mean(u_all * eta)  # scalar

                # Backpropagate and update prediction model
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += loss.detach().cpu().numpy()

            # Early stopping
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy() 
