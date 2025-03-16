import torch
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from metrics import cal_clustering_metric
from sklearn.decomposition import NMF

# 添加注意力层定义
class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2, dropout_rate=None):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # 根据数据稀疏度设置dropout率
        if dropout_rate is None:
            self.dropout_rate = 0.1  # 使用较小的默认值
        else:
            self.dropout_rate = dropout_rate
        
        # 定义可学习的权重矩阵
        self.W = torch.nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = torch.nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        
        # 初始化
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # 添加dropout（训练时使用）
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        
    def forward(self, original_features, gcn_features):
        if self.training:
            # 训练时使用dropout
            Wh = self.dropout(torch.mm(original_features, self.W))
            Wh_gcn = self.dropout(torch.mm(gcn_features, self.W))
        else:
            # 测试时不使用dropout
            Wh = torch.mm(original_features, self.W)
            Wh_gcn = torch.mm(gcn_features, self.W)
        
        # 注意力计算
        a_input = torch.cat([Wh, Wh_gcn], dim=1)
        attention = self.leakyrelu(torch.matmul(a_input, self.a))
        attention = torch.nn.functional.softmax(attention, dim=1)
        
        return attention

class SparseFeaturePooling(torch.nn.Module):
    def __init__(self, orig_dim, gcn_dim, dropout_rate=0.1):
        super(SparseFeaturePooling, self).__init__()
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # 特征压缩层
        self.dim_reduction = torch.nn.Sequential(
            torch.nn.Linear(orig_dim, gcn_dim),
            torch.nn.BatchNorm1d(gcn_dim),
            torch.nn.ReLU()
        )
        
        # 注意力计算
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(gcn_dim * 2, gcn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(gcn_dim, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, original_features, gcn_features):
        # 降维
        orig_reduced = self.dim_reduction(original_features)
        
        if self.training:
            orig_reduced = self.dropout(orig_reduced)
            gcn_features = self.dropout(gcn_features)
        
        # 计算注意力
        combined = torch.cat([orig_reduced, gcn_features], dim=1)
        attention = self.attention(combined)
        
        return attention, orig_reduced

class JLGCN(torch.nn.Module):
    """
        X: n * d
    """

    def __init__(self, X, labels, alpha, beta, d, a, layers=None, acts=None, n_neighbors=None, ro=None, max_epoch=10,
                 max_iter=50,
                 learning_rate=10 ** -2, coeff_reg=10 ** -3,
                 device='cpu'):
        super(JLGCN, self).__init__()
        self.device = device
        self.X = to_tensor(X).to('cpu')
        self.labels = to_tensor(labels).to('cpu')
        self.n_neighbors = n_neighbors
        self.ro = ro
        self.n_clusters = self.labels.unique().shape[0]
        self.alpha = alpha
        self.beta = beta
        if layers is None:
            layers = [32, 16]
        self.layers = layers
        if acts is None:
            layers_count = len(self.layers)
            acts = [torch.nn.functional.relu] * (layers_count - 1)
            acts.append(torch.nn.functional.linear)
        self.acts = acts
        assert len(self.acts) == len(self.layers)
        self.max_iter = max_iter
        self.d = d
        self.a = a
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg
        # X.shape[0]输出X的行数
        # X.shape[1]输出X的列数
        self.data_size = self.X.shape[0]
        self.input_dim = self.X.shape[1]
        self.indicator = None
        self.F = None
        self.embedding = self.X
        self._build_up()
        self.to('cpu')

         # 计算数据稀疏度
        sparsity = (X == 0).sum().item() / (X.shape[0] * X.shape[1])
        
        # 根据稀疏度设置dropout率
        if sparsity > 0.9:
            dropout_rate = 0.05  # 极度稀疏时使用很小的dropout
        elif sparsity > 0.7:
            dropout_rate = 0.1   # 高度稀疏时使用较小的dropout
        else:
            dropout_rate = 0.2   # 正常稀疏度时使用标准dropout
        
        # 添加注意力层
        self.attention_layer = GraphAttentionLayer(
            in_features=X.shape[-1],
            out_features=1,
            dropout_rate=dropout_rate
        ).to(device)
        
        # 添加特征池化层
        self.feature_pooling = SparseFeaturePooling(
            orig_dim=self.input_dim,
            gcn_dim=self.layers[-1],
            dropout_rate=dropout_rate
        ).to(device)
        
        # 添加特征重建层
        self.reconstruction = torch.nn.Linear(self.layers[-1], self.input_dim)
        
        # 添加NMF模型
        self.nmf = NMF(
            n_components=self.n_clusters,
            init='random',
            random_state=42,
            max_iter=200
        )

    def _build_up(self):
        self._gcn_parameters = []
        layers_count = len(self.layers)
        for i in range(layers_count):
            if i is 0:
                self._gcn_parameters.append(get_weight_initial([self.input_dim, self.layers[i]]))
                continue
            self._gcn_parameters.append(get_weight_initial([self.layers[i - 1], self.layers[i]]))
        self._gcn_parameters = torch.nn.ParameterList(self._gcn_parameters)

    def forward(self, Laplacian):
        # 保存原始特征
        original_features = self.X
        
        # GCN前向传播
        embedding = self.X
        for i in range(len(self.layers)):
            embedding = Laplacian.mm(embedding.matmul(self._gcn_parameters[i]))
            if self.acts[i] is not None:
                embedding = self.acts[i](embedding)
        
        # 归一化GCN输出
        epsilon = torch.tensor(10 ** -7).to(self.device)
        gcn_embedding = embedding / embedding.norm(dim=1).reshape((self.data_size, -1)).max(epsilon)
        
        # 特征池化和注意力计算
        attention_scores, orig_reduced = self.feature_pooling(original_features, gcn_embedding)
        
        # 特征融合
        self.embedding = attention_scores * gcn_embedding + (1 - attention_scores) * orig_reduced
        
        # 计算重构矩阵
        recons_A = self.embedding.matmul(self.embedding.t())
        
        # 只返回重构矩阵，不返回reconstructed
        return recons_A

    def __adjacent_mat(self, x, n_neighbors=5):
        """
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        x_np = x.numpy() if torch.is_tensor(x) else x
        A = kneighbors_graph(x_np, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A * np.transpose(A)
        self.adjacency = to_tensor(A)
        dim = self.adjacency.shape[0]
        L = self.adjacency + torch.eye(dim)
        D = L.sum(dim=1)
        sqrt_D = D.pow(-1 / 2)
        Laplacian = sqrt_D * (sqrt_D * L).t()
        self.adjacency = to_tensor(Laplacian)
        return self.adjacency

    def build_loss_reg(self):
        layers_count = len(self.layers)
        loss_reg = 0
        for i in range(layers_count):
            loss_reg += self._gcn_parameters[i].abs().sum()
            # loss_reg += self._gcn_parameters[i].norm()**2
        return loss_reg

    def build_pretrain_loss(self, recons_A):
        # 确保recons_A是张量而不是元组
        if isinstance(recons_A, tuple):
            recons_A, reconstructed = recons_A
        
        # 原有的损失计算
        recons_A = recons_A - recons_A.diag().diag()
        return torch.sum(torch.pow(recons_A, 2)) / (self.data_size ** 2)

    def update_indicator(self, features):
        if features.requires_grad:
            features = features.detach()
        
        # 确保特征非负
        features_np = features.cpu().numpy()
        features_np = np.abs(features_np)
        
        try:
            # 进行NMF分解
            # V ≈ WH, 其中:
            # V ∈ ℝ^(n×m): 原始特征矩阵
            # W ∈ ℝ^(n×k): 基矩阵
            # H ∈ ℝ^(k×m): 系数矩阵
            _ = self.nmf.fit_transform(features_np)
            H = self.nmf.components_  # 获取H矩阵
            
            # 将H矩阵转置后作为indicator
            # H^T ∈ ℝ^(m×k)
            H_t = H.T
            self.indicator = torch.from_numpy(H_t).float()
        except:
            print('NMF Not Convergence')
        
        self.indicator = self.indicator.detach()

    def clustering(self):
        if self.indicator.requires_grad:
            indicator = self.indicator.detach()
        else:
            indicator = self.indicator
        
        # 转换为numpy进行后续处理
        H_t = indicator.cpu().numpy()  # H^T ∈ ℝ^(m×k)
        
        # 使用H矩阵的列向量进行聚类
        # 每个特征(行)分配给最大权重的主题(列)
        grps = np.argmax(np.abs(H_t), axis=1).astype(np.int32)
        
        # 计算聚类指标
        acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), grps)
        
        return acc, nmi, ari, f1, grps, H_t

    def build_loss(self, recons_A, C_final, reconstructed=None):
        # 原有的损失项
        loss_1 = self.build_pretrain_loss(recons_A)
        
        # 重建损失（针对稀疏数据的masked MSE）
        if reconstructed is not None:
            mask = (self.X != 0).float()  # 只考虑非零元素
            recon_loss = torch.sum(mask * (self.X - reconstructed)**2) / torch.sum(mask)
            loss_1 = loss_1 + 0.1 * recon_loss  
        
        # 最终，将所有元素加和并除以节点数的平方，得到邻接矩阵重构损失loss_1
        loss_1 = loss_1.sum() / (self.data_size ** 2)
        a_1 = self.X.t().matmul(self.adjacency)
        C_final = C_final.float()
        loss_21 = a_1.matmul(C_final) - self.X.t()
        loss_21 = loss_21.norm() ** 2 / (loss_21.shape[0] * loss_21.shape[1])
        loss_22 = C_final.norm() ** 2 / (C_final.shape[0] * C_final.shape[1])
        self.indicator = self.indicator.float()
        loss_3 = C_final.t() - C_final.t().matmul(self.indicator).matmul(self.indicator.t())
        loss_3 = loss_3.norm() ** 2 / (loss_3.shape[0] * loss_3.shape[1])
        loss_reg = self.build_loss_reg()
        loss = loss_1 + 1 / 2 * loss_21 + self.alpha / 2 * loss_22 + self.beta * loss_3 + self.coeff_reg * loss_reg
        return loss

    def pretrain(self, pretrain_iter, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        print('Start pretraining (totally {} iterations) ......'.format(pretrain_iter))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        Laplacian = self.__adjacent_mat(self.X, self.n_neighbors)

        for i in range(pretrain_iter):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            loss = self.build_pretrain_loss(recons_A)
            loss.backward()
            optimizer.step()
        print(loss.item())

    def run(self):
        f = 1
        objs = torch.zeros((250, 1), dtype=torch.float64)
        a_1 = np.zeros((50, 4))
        a_2 = np.zeros((4, 4))
        a_3 = self.labels
        a_4 = self.adjacency
        m = 0
        C_final, flag = self.lrr()
        # localX_np = C_final.cpu().numpy()  # 将float32类型的张量localX转化为numpy数组
        # min_neighbour = min(np.sum(localX_np != 0, axis=0))  # 计算非零元素的个数并取其中的最小值
        # n=C_final.shape[0]

        # if flag and ((n<1000 and min_neighbour>3) or (n>=1000 and min_neighbour>1)):
        if flag:

            self.update_indicator(C_final)
            # acc, nmi, ari, f1,flag = self.clustering()
            acc, nmi, ari, f1, lab, sim = self.clustering()
            print('Initial ACC: %.2f, NMI: %.2f, ARI: %.2f' % (acc * 100, nmi * 100, ari * 100))
            objs = []
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            Laplacian = get_Laplacian(self.adjacency)

            for epoch in range(self.max_epoch):
                assert not self.indicator.requires_grad
                for i in range(self.max_iter):
                    optimizer.zero_grad()
                    recons_A = self(Laplacian)
                    loss = self.build_loss(recons_A, C_final)
                    loss.backward()
                    optimizer.step()
                    objs.append(loss.item())
                    # print('loss: ', loss.item())
                C_final, flag = self.lrr()
                localX_np = C_final.cpu().numpy()  # 将float32类型的张量localX转化为numpy数组
                min_neighbour = min(np.sum(localX_np != 0, axis=0))  # 计算非零元素的个数并取其中的最小值
                n = C_final.shape[0]
                if flag and ((n < 1000 and min_neighbour > 3) or (n >= 1000 and min_neighbour > 1)):
                # if flag:
                    self.update_indicator(C_final)
                    acc, nmi, ari, f1, lab, sim = self.clustering()
                    loss = self.build_loss(recons_A, C_final)
                    objs.append(loss.item())
                    print('loss: %.4f, ACC: %.2f, NMI: %.2f, ARI: %.2f, F1: %.2f' % (
                        loss.item(), acc * 100, nmi * 100, ari * 100, f1 * 100))
                    a_1[epoch, 0] = acc
                    a_1[epoch, 1] = nmi
                    a_1[epoch, 2] = ari
                    a_1[epoch, 3] = f1
                    if m < nmi:
                        m = nmi
                        a_3 = lab
                        a_4 = sim
                else:
                    f = 0
                    return np.array(objs), np.array(a_1), np.array(a_3), np.array(a_4), np.array(a_2), f
            a_2[0] = a_1.mean(axis=0)
            a_2[1] = a_1.var(axis=0)
            a_2[2] = a_1.max(axis=0)
            a_2[3] = a_1.min(axis=0)

            return np.array(objs), np.array(a_1), np.array(a_3), np.array(a_4), np.array(a_2), f
        else:
            f = 0
            return np.array(objs), np.array(a_1), np.array(a_3), np.array(a_4), np.array(a_2), f

    def lrr(self):
        flag = 1

        A = self.embedding.matmul(self.embedding.t())
        X_ = torch.from_numpy(self.X.numpy().T)
        X_embedding = torch.matmul(X_, A)
        I = torch.eye(X_embedding.shape[1], dtype=torch.float32)
        X_transpose = torch.transpose(X_embedding, 0, 1)
        AA = torch.matmul(X_transpose, X_embedding) + self.alpha * I
        det = torch.det(AA)
        if abs(det) < 1e-8:
            flag = 0
            return I, flag
        else:
            inv = torch.inverse(AA)
            AAA = torch.matmul(inv, X_transpose)
            C = torch.matmul(AAA, X_)
            Coef = self.thrC(C, self.ro)
            C_final = self.post_proC(Coef, self.n_clusters, self.d, self.a)
            return C_final, flag


    def post_proC(self, C, K, d, alpha):
        # 确保C不需要梯度
        if C.requires_grad:
            C = C.detach()
        
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = torch.linalg.svd(C)
        U, S = U[:, :r], S[:r]
        U = torch.flip(U, dims=(1,))
        S = torch.sqrt(torch.flip(S, dims=(0,)))
        S = torch.diag(S)
        U = U @ S
        
        # 确保在转换为numpy之前detach
        if U.requires_grad:
            U = U.detach()
        U_numpy = U.cpu().numpy()
        
        # 后续操作
        U_normalized = normalize(U_numpy, norm='l2', axis=1)
        Z = U_normalized @ U_normalized.T
        Z = Z * (Z > 0)
        L = torch.from_numpy(Z).float()
        L = torch.abs(L ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        return L

    def thrC(self, C, ro):
        if ro[0] < 1:
            N = C.shape[1]
            Cp = torch.zeros((N, N), dtype=torch.float32).to('cpu')
            S = torch.abs(torch.sort(-torch.abs(C), dim=0).values)
            Ind = torch.argsort(-torch.abs(C), dim=0)
            for i in range(N):
                cL1 = torch.sum(S[:, i]).type(torch.float32).to('cpu')
                stop = False
                csum = 0
                t = 0
                while (stop == False):

                    csum = csum + S[t, i]
                    if csum > ro[0] * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


def to_tensor(X):
    if type(X) is torch.Tensor:
        return X
    return torch.Tensor(X)


def get_Laplacian(A):
    dim = A.shape[0]
    L = A + torch.eye(dim)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-1 / 2)
    Laplacian = sqrt_D * (sqrt_D * L).t()
    return Laplacian