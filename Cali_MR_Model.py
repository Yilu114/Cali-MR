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
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users # user数
        self.num_items = num_items # item数
        self.embedding_k = embedding_k # 隐向量维数
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation1 = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation2 = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model1 = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model2 = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        
        self.propensity_soft_binning = nn.Linear(in_features=1, out_features=10)
        self.imputation_soft_binning = nn.Linear(in_features=1, out_features=10)

        self.alpha = torch.nn.Parameter(torch.randn(2)) # propensity model的linear系数
        self.beta = torch.nn.Parameter(torch.randn(2)) # imputation model的linear系数

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def _compute_IPS(self, x, batch_size_prop = 2048,
        num_epoch=1000, lr=0.05, lamb=0, gamma=1,
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # 对于coat而言，obs的形状是(87000, 1)，是由290*300这个observation matrix做reshap -1得到
        optimizer_propensity1 = torch.optim.Adam(self.propensity_model1.parameters(), lr=lr, weight_decay=lamb)
        optimizer_propensity2 = torch.optim.Adam(self.propensity_model2.parameters(), lr=lr, weight_decay=lamb)
        optimizer_propensity_soft_binning = torch.optim.Adam(self.propensity_soft_binning.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items) # (87000， 2) shape，所有可能的序号合集
        early_stop = 0
    
        w_prop = F.softmax(self.alpha, dim=0)
        optimizer_w = torch.optim.Adam([self.alpha], lr=lr, weight_decay=lamb)

        for epoch in range(num_epoch):

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * batch_size_prop : (idx+1) * batch_size_prop] #样本序号
                
                x_sampled = x_all[x_all_idx] # 样本

                sub_obs = obs[x_all_idx] # 样本对应的oui
                sub_obs = torch.Tensor(sub_obs).cuda()

                prop1 = self.propensity_model1.forward(x_sampled).squeeze() # 样本对应的预测propensity 1
                prop2 = self.propensity_model2.forward(x_sampled).squeeze() # 样本对应的预测propensity 2

                # assumed update w
                prop = w_prop[0] * prop1 + w_prop[1] * prop2
                loss_w = F.binary_cross_entropy(prop, sub_obs)

                alpha_grad = torch.autograd.grad(loss_w, ([self.alpha]), create_graph=True)[0]
                updated_alpha = self.alpha - optimizer_w.param_groups[0]['lr'] * alpha_grad
                # 伪更新linear的系数，伪更新的含义，是指我们在数值上更新一步，但是并不真正更新参数

                # print("Original alpha:", self.alpha)
                # print("Updated alpha:", updated_alpha)


                # true update prop
                prop_loss_1 = F.binary_cross_entropy(prop1, sub_obs, reduction="mean")
                prop_loss_2 = F.binary_cross_entropy(prop2, sub_obs, reduction="mean")

                updated_w_prop = F.softmax(updated_alpha, dim=0)
                updated_prop = (updated_w_prop[0] * prop1 + updated_w_prop[1] * prop2).view(-1, 1)
                # w1p1 + w2p2

                prop_soft_binning = F.softmax(self.propensity_soft_binning.forward(updated_prop), dim=-1)

                # print("1")
                # print(prop_soft_binning.shape)
                # print(sub_obs.shape)
                # print(sub_obs.unsqueeze(1).sum(dim=0).shape)
                # print(prop_soft_binning.sum(dim=0).shape)

                acc_m = torch.sum(prop_soft_binning * sub_obs.unsqueeze(1), dim=0)
                conf_m = torch.sum(prop_soft_binning * updated_prop, dim=0)
                prop_dece_m = torch.abs(acc_m - conf_m) # 每个桶里的校准损失，我们实验里是10个桶
  
                prop_dece_loss = torch.mean(prop_dece_m) # 总体校准损失


                loss_prop = prop_loss_1 + prop_loss_2 + gamma * prop_dece_loss

                # print("1")
                # print(prop_loss_1)
                # print(prop_loss_2)
                # print(prop_dece_loss)

                optimizer_propensity1.zero_grad()
                optimizer_propensity2.zero_grad()
                loss_prop.backward(retain_graph=True)
                optimizer_propensity1.step()
                optimizer_propensity2.step()


                # true update w
                prop1 = self.propensity_model1.forward(x_sampled).squeeze()
                prop2 = self.propensity_model2.forward(x_sampled).squeeze()

                prop = w_prop[0] * prop1 + w_prop[1] * prop2
                loss_w = F.binary_cross_entropy(prop, sub_obs)

                optimizer_w.zero_grad()
                loss_w.backward(retain_graph=True)
                optimizer_w.step()

                # true update soft binning
                w_prop = F.softmax(self.alpha, dim=0) 
                prop =  (w_prop[0] * prop1 + w_prop[1] * prop2).view(-1, 1)

                prop_soft_binning = F.softmax(self.propensity_soft_binning.forward(prop), dim=-1)

                acc_m = torch.sum(prop_soft_binning * sub_obs.unsqueeze(1), dim=0)
                conf_m = torch.sum(prop_soft_binning * prop, dim=0)
                prop_dece_m = torch.abs(acc_m - conf_m)
  
                prop_dece_loss = torch.mean(prop_dece_m)

                optimizer_propensity_soft_binning.zero_grad()
                prop_dece_loss.backward(retain_graph=True)
                optimizer_propensity_soft_binning.step()

                
                epoch_loss += prop_loss_1.detach().cpu().numpy() + prop_loss_2.detach().cpu().numpy() + prop_dece_loss.detach().cpu().numpy()


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


        
    def fit(self, x, y, unlabel_x, stab = 0,
        num_epoch=1000, batch_size=128, lr1=0.05, lamb1=0, lr2=0.05, lamb2 = 0, lr3=0.05, lamb3 = 0, gamma=1, prop_clip =0.05,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
        optimizer_imputation1 = torch.optim.Adam(
            self.imputation1.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_imputation2 = torch.optim.Adam(
            self.imputation1.parameters(), lr=lr2, weight_decay=lamb2)
        optimizer_imputation_soft_binning = torch.optim.Adam(
            self.imputation_soft_binning.parameters(), lr=lr3, weight_decay=lamb3)
                      
        last_loss = 1e9

        num_sample = len(x) 
        total_batch = num_sample // batch_size

        v_imp = F.softmax(self.beta, dim=0) 
        optimizer_v = torch.optim.Adam([self.beta], lr=lr1, weight_decay=lamb1)

        w_prop = F.softmax(self.alpha, dim=0) 
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(unlabel_x.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                # prop
                prop1 = self.propensity_model1.forward(sub_x).squeeze()
                prop2 = self.propensity_model2.forward(sub_x).squeeze()
                prop = w_prop[0] * prop1 + w_prop[1] * prop2
                inv_prop = 1 / torch.clip(prop, prop_clip, 1).detach() 

                # assumed update 系数
                imp1 = self.imputation1.forward(sub_x).squeeze()
                imp2 = self.imputation2.forward(sub_x).squeeze()
                imp_y = v_imp[0] * imp1 + v_imp[1] * imp2

                pred = self.prediction_model.predict(sub_x).cuda()
                sub_y = torch.Tensor(sub_y).cuda()

                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")

                e_hat_loss = F.binary_cross_entropy(imp_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()

                alpha_grad = torch.autograd.grad(imp_loss, ([self.beta]), create_graph=True)[0]
                updated_beta = self.beta - optimizer_v.param_groups[0]['lr'] * alpha_grad

                # print("Original beta:", self.beta)
                # print("Updated beta:", updated_beta)

                # true update imputation
                e_hat_loss1 = F.binary_cross_entropy(imp1, pred, reduction="none")
                imp_loss_1 = (((e_loss - e_hat_loss1) ** 2) * inv_prop).mean()

                e_hat_loss2 = F.binary_cross_entropy(imp2, pred, reduction="none")
                imp_loss_2 = (((e_loss - e_hat_loss2) ** 2) * inv_prop).mean()

                updated_v_imp = F.softmax(updated_beta, dim=0) 
                updated_imp_y = (updated_v_imp[0] * imp1 + updated_v_imp[1] * imp2).view(-1, 1)

                imp_soft_binning = F.softmax(self.imputation_soft_binning.forward(updated_imp_y), dim=-1)

                # print(imp_soft_binning.shape)
                # print(e_loss.shape)
                # print(inv_prop.shape)
                # print(imp_soft_binning.sum(dim=0).shape)

                acc_m = torch.sum(imp_soft_binning * e_loss.unsqueeze(1), dim=0) / torch.sum(inv_prop)
                conf_m = torch.sum(imp_soft_binning * updated_imp_y, dim=0)

                imp_dece_m = torch.abs(acc_m - conf_m) 
                imp_dece_loss = torch.mean(imp_dece_m)

                loss_imp = imp_loss_1 + imp_loss_2 + gamma * imp_dece_loss

                optimizer_imputation1.zero_grad()
                optimizer_imputation2.zero_grad()
                loss_imp.backward(retain_graph=True)
                optimizer_imputation1.step()
                optimizer_imputation2.step()


                # true update v
                imp1 = self.imputation1.forward(sub_x).squeeze()
                imp2 = self.imputation2.forward(sub_x).squeeze()
                imp_y = v_imp[0] * imp1 + v_imp[1] * imp2

                e_hat_loss = F.binary_cross_entropy(imp_y, pred, reduction="none")
                loss_v = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_v.zero_grad()
                loss_v.backward(retain_graph=True)
                optimizer_v.step()


                # true update soft binning
                v_imp = F.softmax(self.beta, dim=0) 
                imp_y = (v_imp[0] * imp1 + v_imp[1] * imp2).view(-1, 1)

                imp_soft_binning = F.softmax(self.imputation_soft_binning.forward(imp_y), dim=-1)

                acc_m = torch.sum(imp_soft_binning * e_loss.unsqueeze(1), dim=0) / torch.sum(inv_prop)
                conf_m = torch.sum(imp_soft_binning * imp_y, dim=0)

                imp_dece_m = torch.abs(acc_m - conf_m) 
                imp_dece_loss = torch.mean(imp_dece_m)


                optimizer_imputation_soft_binning.zero_grad()
                imp_dece_loss.backward(retain_graph=True)
                optimizer_imputation_soft_binning.step()


                # update prediction model
                x_sampled = unlabel_x[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                inv_prop1 = 1 / torch.clip(self.propensity_model1.forward(sub_x), prop_clip, 1).detach() 
                inv_prop2 = 1 / torch.clip(self.propensity_model2.forward(sub_x), prop_clip, 1).detach()  
             
                pred = self.prediction_model.forward(sub_x)
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                u = torch.Tensor(np.c_[inv_prop1.cpu().numpy(), inv_prop2.cpu().numpy(), self.imputation1.predict(sub_x).numpy(), self.imputation2.predict(sub_x).numpy()])
                matrix_inv = torch.Tensor(np.linalg.inv(u.T.matmul(u) + stab * torch.eye(4))).cuda()

                eta = torch.squeeze(matrix_inv.matmul(u.T.cuda().matmul(e_loss)))
            
                
                inv_prop_all1 = 1 / torch.clip(self.propensity_model1.forward(x_sampled), prop_clip, 1).detach().cpu().numpy() 
                inv_prop_all2 = 1 / torch.clip(self.propensity_model2.forward(x_sampled), prop_clip, 1).detach().cpu().numpy() 
                u = torch.Tensor(np.c_[inv_prop_all1, inv_prop_all2, self.imputation1.predict(x_sampled).numpy(), self.imputation2.predict(x_sampled).numpy()]).cuda()
                
                loss = torch.mean(u * eta)
                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()                   
                
                epoch_loss += loss.detach().cpu().numpy()
                
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




