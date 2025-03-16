# from data_loader import *
# ... existing code ...
import scipy.io as scio  # 添加这行导入语句
# ... existing code ...


from data_loader import *
from renew_scgae import JLGCN
import torch
import warnings
import os
import numpy as np
import random

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 设置随机种子以确保结果可重复
    seed = 42  # 可以选择任意整数作为种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    datasets = [
        'Quake_Smart-seq2_Diaphragm',
        'Quake_Smart-seq2_Limb_Muscle',
        'Quake_Smart-seq2_Trachea',
        'Quake_Smart-seq2_Lung',
        'Muraro',
        'Quake_10x_Bladder',
        'Romanov',
        'Adam',
        'Young',
        'Quake_10x_Spleen'
    ]

    layers = [256, 128]
    acts = [torch.nn.functional.relu] * len(layers)
    n_neighbors = 5
    ro = 0.5,

    learning_rate = 10**-4*4
    pretrain_learning_rate = 0.001
    dataset_path = r'C:\\123\\zh\\code\\scgae_code\\result\\tiaocan'
  
    for pretrain_epochs in range(10, 110, 10):  # 从10到100,步长为10    
        for coeff_reg in [0.001]:
            for i in range(8,9):
                name = datasets[i]
                current_dataset_path = os.path.join(dataset_path, name)      
                features, labels = process_all_datasets(name)
                AAA = 0
                aa = labels
                alpha_1 = 0
                beta_1 = 0
                d_1 = 0
                a_1 = 0
                losses_1 = 0
                pretrain_epochs_1 = 0  # 添加变量存储最佳预训练轮数
                ac_1 = np.zeros((50, 4))
                var_avg_best = np.zeros((4, 4))
                s = torch.eye(features.shape[0], dtype=torch.float32)

                
                for alpha in (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000):
                    for beta in (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000):
                        for d in [4]:
                            for a in [4]:
                                print('==========dataname={},pretrain_epochs={}, alpha={},beta={},d={}, a={}, reg={} =========='.format(
                                    name,pretrain_epochs, alpha, beta, d, a, coeff_reg))

                                
                                model = JLGCN(
                                    features, labels, alpha, beta, d, a,
                                    layers=layers, acts=acts,
                                    n_neighbors=n_neighbors, ro=ro,
                                    max_epoch=50, max_iter=4,
                                    coeff_reg=coeff_reg,
                                    learning_rate=learning_rate
                                )

                                flag = model.pretrain(pretrain_epochs, learning_rate=pretrain_learning_rate)
                                if flag == 0:
                                    continue
                                else:
                                    losses, ac, lab, sim, var_avg, f = model.run()
                                    if f == 0:
                                        break
                                    else:
                                        if AAA < var_avg[0, 1]:
                                            AAA = var_avg[0, 1]
                                            aa = lab
                                            s = sim
                                            alpha_1 = alpha
                                            beta_1 = beta
                                            a_1 = a
                                            d_1 = d
                                            pretrain_epochs_1 = pretrain_epochs  # 记录最佳预训练轮数
                                            ac_1 = ac
                                            losses_1 = losses
                                            var_avg_best = var_avg
                                if not os.path.exists(current_dataset_path):
                                    os.makedirs(current_dataset_path)
                                # 保存中间结果
                                save_name = f'losses_{name}_alpha{alpha}_beta{beta}_d{d}_a{a}_epochs{pretrain_epochs}.mat'
                                save_path = os.path.join(current_dataset_path, save_name)
                                
                                scio.savemat(save_path,
                                            {
                                            'alpha': alpha, 'beta': beta, 'd': d, 'a': a,
                                            'pretrain_epochs': pretrain_epochs,
                                            'losses': losses, 'ac': ac, 'lab': lab, 'sim': sim,
                                            'var_avg': var_avg, 'var_avg_best': var_avg_best,
                                            'label_best': aa, 'sim_best': s,
                                            'alpha_best': alpha_1, 'beta_best': beta_1,
                                            'a_best': a_1, 'd_best': d_1,
                                            'pretrain_epochs_best': pretrain_epochs_1,  # 保存最佳预训练轮数
                                            'losses_best': losses_1, 'ac_best': ac_1})

            # 保存最终结果
            final_save_name = f'losses_{name}_final_results.mat'
            final_save_path = os.path.join(current_dataset_path, final_save_name)
            
            scio.savemat(final_save_path,
                        {
                         'alpha': alpha_1, 'beta': beta_1, 'd': d_1, 'a': a_1,
                         'pretrain_epochs': pretrain_epochs_1,  # 保存最佳预训练轮数
                         'losses': losses_1, 'ac': ac_1, 'lab': aa, 'sim': s,
                         'var_avg_best': var_avg_best})





