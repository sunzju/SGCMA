import torch


def kl_multivariate_normal(mu0, sigma0, mu1, sigma1):  
    """  
    计算两个多元高斯分布 N(mu0, sigma0) 与 N(mu1, sigma1) 的 KL 散度  
    mu0, mu1:      (..., d)       均值向量  
    sigma0, sigma1:(..., d, d)    协方差矩阵（必须对称正定）  

    返回:           (...)          KL散度  
    """  
    d = mu0.shape[-1]  

    # 计算 Sigma1 的逆与对数行列式，Sigma0 的对数行列式  
    inv_sigma1 = torch.linalg.inv(sigma1)  
    logdet_sigma0 = torch.linalg.slogdet(sigma0)[1]  
    logdet_sigma1 = torch.linalg.slogdet(sigma1)[1]  

    # 第一项 tr(Sigma1^-1 Sigma0)  
    trace_term = torch.einsum('...ij,...jk->...ik', inv_sigma1, sigma0).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  

    # 第二项 (mu1-mu0)^T Sigma1^-1 (mu1-mu0)  
    diff = (mu1 - mu0)[..., None]          # (..., d, 1)  
    mahalanobis = torch.matmul(torch.matmul(diff.transpose(-2,-1), inv_sigma1), diff).squeeze(-1).squeeze(-1)  

    # KL公式汇总  
    kl = 0.5 * (trace_term + mahalanobis - d + logdet_sigma1 - logdet_sigma0)  
    return kl  
def compute_metrics(out_logits, targets):
    out_labels = torch.argmax(out_logits, dim=1)
    num_classes = out_logits.size(-1)
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    for c in range(num_classes):
        true_positives = torch.sum((out_labels == targets) & (targets == c))
        predicted_positives = torch.sum(out_labels == c)
        actual_positives = torch.sum(targets == c)

        precision = true_positives.float() / predicted_positives if predicted_positives else 0
        recall = true_positives.float() / actual_positives if actual_positives else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        accuracy = true_positives.float() / actual_positives if actual_positives else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
    precision = torch.mean(torch.tensor(precision_list))
    recall = torch.mean(torch.tensor(recall_list))
    f1 = torch.mean(torch.tensor(f1_list))
    accuracy = torch.mean(torch.tensor(accuracy_list))
    return precision, recall, f1, accuracy


def KL_local_global(T_o_local, T2_o_local, T_n_local, T2_n_local, T_o_other_sum, T2_o_other_sum, T_n_other, T2_n_other, sample_num_local, sample_num_class_others):
    """
    ::param T_o_local: [num_classes, z_o_dim]
    ::param T2_o_local: [num_classes, z_o_dim, z_o_dim]
    ::param T_n_local: [z_n_dim]
    ::param T2_n_local: [z_n_dim, z_n_dim]
    ::param T_o_other_sum: [num_classes, z_o_dim]
    ::param T2_o_other_sum: [num_classes, z_o_dim, z_o_dim]
    ::param T_n_other: [client_num-1, z_n_dim]
    ::param T2_n_other: [client_num-1, z_n_dim, z_n_dim]
    ::param sample_num_local: [num_classes] 本地每个类别的样本数量
    ::param sample_num_class_others: [client_num-1, num_classes] 其他客户端每个类别的样本数量
    """

    num_classes = len(T_o_local)
    z_o_dim = T_o_local.size(1)
    z_n_dim = T_n_local.size(0)
    client_num = len(T_n_other)
    sample_num_others = sample_num_class_others.sum(dim=0)
    sample_nums = sample_num_local + sample_num_others
    T_o = (T_o_local + T_o_other_sum) / sample_nums.unsqueeze(1)
    T2_o = (T2_o_local + T2_o_other_sum) / sample_nums.unsqueeze(1).unsqueeze(2)

    mu_o = T_o
    Sigma_o = T2_o - T_o.unsqueeze(2) @ T_o.unsqueeze(1) 
    # Sigma_o = Sigma_o + 1e-6 * torch.eye(z_o_dim)
    # Sigma_o_inv = torch.linalg.inv(Sigma_o)
    # log_det_Sigma_o = torch.slogdet(Sigma_o)[0]

    # tr_Sigma2_inv_Simga_1 = torch.einsum('mcd,ndf->mncf', Sigma_o_inv, Sigma_o).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) # [m, n, c, f] -> [m, n]
    # mu2_mu1 = mu_o.unsqueeze(1) - mu_o.unsqueeze(0) # [m, n, d]

    # mu2_mu1_T_Simga2_inv = torch.einsum('mnd,mdc->mnc', mu2_mu1, Sigma_o_inv) # [m, n, c]
    # mu2_mu1_T_Simga2_inv_mu2_mu1 = torch.einsum('mnc,mnc->mn', mu2_mu1_T_Simga2_inv, mu2_mu1) # [m, n]

    # log_det_Sigma1_log_det_Sigma2 = log_det_Sigma_o.unsqueeze(0) - log_det_Sigma_o.unsqueeze(1)

    # interclass_kl_sum = 0.5 * (tr_Sigma2_inv_Simga_1 +mu2_mu1_T_Simga2_inv_mu2_mu1 + log_det_Sigma1_log_det_Sigma2 - num_classes)

    # mask = 1 - torch.eye(num_classes)
    # interclass_kl_sum = interclass_kl_sum * mask.unsqueeze(-1)
    # inter_clas_kl_loss = torch.sum(interclass_kl_sum) / (num_classes*(num_classes-1))

    inter_clas_kl_loss = 0
    for c in range(num_classes):
        mu_o_c = mu_o[c]
        Sigma_o_c = Sigma_o[c]
        for c_other in range(num_classes):
            if c == c_other:
                continue
            mu_o_c_other = mu_o[c_other]
            Sigma_o_c_other = Sigma_o[c_other]
            kl_loss = kl_multivariate_normal(mu_o_c, Sigma_o_c, mu_o_c_other, Sigma_o_c_other)
            inter_clas_kl_loss += kl_loss

    return inter_clas_kl_loss/(num_classes*(num_classes-1))


if __name__ == "__main__":
    num_class = 10
    feature_dim = 1024
    client_num = 5
    sample_num_local = torch.ones(num_class).cuda() * 32
    sample_num_class_others = torch.ones(client_num-1, num_class).cuda() * 32

    zo = torch.randn(num_class, 32, feature_dim).cuda() 
    zo2 = zo[:, :, :, None] @ zo[:, :, None, :]



    T_o_local = zo.sum(dim=-2).cuda()
    T2_o_local = zo2.sum(dim=-3).cuda()
    T_n_local = torch.randn(feature_dim).cuda()
    T2_n_local = torch.randn(feature_dim, feature_dim).cuda()
    T_o_other_sum = T_o_local * (client_num-1)
    T2_o_other_sum = T2_o_local * (client_num-1)
    T_n_other = torch.randn(client_num-1, feature_dim).cuda()
    T2_n_other = torch.randn(client_num-1, feature_dim, feature_dim).cuda()

    inter_clas_kl_loss = KL_local_global(T_o_local, T2_o_local, T_n_local, T2_n_local, T_o_other_sum, T2_o_other_sum, T_n_other, T2_n_other, sample_num_local, sample_num_class_others)
    print(inter_clas_kl_loss)
    # d = 3  
    # mu0 = torch.zeros(d)  
    # sigma0 = torch.eye(d)  
    # mu1 = torch.zeros(d)  
    # sigma1 = torch.eye(d)  

    # kl = kl_multivariate_normal(mu0, sigma0, mu1, sigma1)  
    # print(f"KL 散度结果：{kl:.4f}")  