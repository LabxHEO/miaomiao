import numpy as np
import scipy.sparse as sp
import h5py
from sklearn.preprocessing import normalize
import os

def load_h5_data(file_path):
    """
    读取h5格式的单细胞数据
    """
    with h5py.File(file_path, 'r') as f:
        # 读取表达矩阵
        data = f['exprs/data'][:]
        indices = f['exprs/indices'][:]
        indptr = f['exprs/indptr'][:]
        shape = f['exprs/shape'][:]
        X = sp.csr_matrix((data, indices, indptr), shape=shape)
        
        # 读取细胞类型标签
        labels = f['obs/cell_type1'][:]
        # 将字节字符串转换为普通字符串
        labels = np.array([label.decode('utf-8') for label in labels])
        
        # 基因过滤
        m = X.shape[0]
        flag = (X != 0)
        flag_count = np.array(flag.sum(axis=0)).flatten()
        flag_index = flag_count > (m * 0.05)
        X = X[:, flag_index]

        # 归一化
        X = normalize(X)

        # 处理 NaN 值
        X = X.todense()
        indices = np.where(np.isnan(X))
        X[indices] = 0

        # 将字符串标签转换为数值标签
        unique_labels = np.unique(labels)
        label_dict = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_dict[label] for label in labels])

        return X, numeric_labels

def process_all_datasets(name):
    """
    处理单个数据集
    """
    try:
        print(f"Processing {name}...")
        path = r'C:\\123\\zh\\code\\dataset\\{}\\data.h5'.format(name)
        
        with h5py.File(path, 'r') as f:
            # 读取表达矩阵
            data = f['exprs/data'][:]
            indices = f['exprs/indices'][:]
            indptr = f['exprs/indptr'][:]
            shape = f['exprs/shape'][:]
            X = sp.csr_matrix((data, indices, indptr), shape=shape)
            
            # 读取细胞类型标签
            labels = f['obs/cell_type1'][:]
            # 将字节字符串转换为普通字符串
            labels = np.array([label.decode('utf-8') for label in labels])
            
            # 基因过滤
            m = X.shape[0]
            flag = (X != 0)
            flag_count = np.array(flag.sum(axis=0)).flatten()
            flag_index = flag_count > (m * 0.05)
            X = X[:, flag_index]

            # 归一化
            X = normalize(X)

            # 处理 NaN 值
            X = X.todense()
            indices = np.where(np.isnan(X))
            X[indices] = 0

            # 将字符串标签转换为数值标签
            unique_labels = np.unique(labels)
            label_dict = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_dict[label] for label in labels])

            print(f"Successfully processed {name}. Shape: {X.shape}, Number of classes: {len(np.unique(numeric_labels))}")
            return X, numeric_labels
            
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")
        return None, None

if __name__ == '__main__':
    results = process_all_datasets()
    
    # 打印处理结果摘要
    print("\nProcessing Summary:")
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"Data shape: {data['shape']}")
        print(f"Number of classes: {data['num_classes']}")
        print(f"Memory usage: {data['X'].nbytes / 1024 / 1024:.2f} MB")