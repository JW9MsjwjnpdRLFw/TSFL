import os
import pickle
from utlis.Utils import *
from scipy.sparse import csr_matrix
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy import signal

class Adjaency_Generator:

    def __init__(self,mode = "plv",threshold = None):
        if mode not in ["distance","knn","pcc","plv"]:
            assert 0,"Not supporting adjacency generate mode"
        self.mode = mode
        if threshold:
            self.threshold = threshold

    def get_adj(self,length_list,feature, path):
        if self.mode == "distance":
            process_adj(length_list, path)
        if self.mode == "knn":
            process_adj_knn(feature, path)
        if self.mode == "pcc":
            process_adj_pcc(feature, path)
        if self.mode == "plv":
            process_adj_plv(feature, path)


def preprocess_label(path):
    """

    :param label_list:
    save:8598 x 5 (graph number x category number)
    :return:
    """
    ReadList = np.load(path['feature'], allow_pickle=True)
    a = ReadList['train_targets']
    b = ReadList['val_targets']
    np.save(path['save']+"labels.npy",np.concatenate((a,b)))
    print("label:{}".format(np.concatenate((a,b)).shape[0]))

    return

def preprocess_feature(path):
    """

    save:[array<node_number x feature embedding>] length: graph number
    :return:
    """
    ReadList = np.load(path['feature'], allow_pickle=True)
    a = ReadList['train_feature']
    b = ReadList['val_feature']
    c = np.concatenate((a,b))
    print("feature:{}".format(c.shape[0]))


    rlt = [ele for ele in c]
    output = open(path['save']+"feature_matrices.pkl","wb")
    pickle.dump(rlt,output)

    return rlt

def process_adj(length, path):
    """
    save:[csr_matrix] length: graph number
    :return:
    """
    print("adj:{}".format(np.sum(length)))
    Dis_Conn = np.load(path['disM'], allow_pickle=True)  # shape:[V,V]
    L_DC = scaled_Laplacian(Dis_Conn)                    # Calculate laplacian matrix
    rlt = [csr_matrix(L_DC) for _ in range(np.sum(length))]
    output = open(path['save']+"adjacency_matrices.pkl","wb")
    pickle.dump(rlt,output)
    print("finished adj")


def process_adj_knn(feature, path):
    adj_generate = Adjacency(n_neighbors= 5)
    rlt = []
    for i in tqdm(range(len(feature))):
        graph = feature[i]
        rlt.append(adj_generate.create_adjacency(graph))
    output = open(path['save']+"adjacency_matrices.pkl","wb")
    pickle.dump(rlt,output)
    print("finished adj")


class pearson_adj:

    def __init__(self,threshold = None):
        if threshold != None:
            self.threshold = threshold
        else:
            self.threshold = 0

    def create_adjacency(self,graph):
        node_num = graph.shape[0]
        adj_init = np.eye(node_num,node_num)
        for i in range(node_num):
            for j in range(node_num):
                pear ,_= pearsonr(graph[i],graph[j])
                if  pear > self.threshold:
                    adj_init[i][j] = pear
        return csr_matrix(adj_init)

def process_adj_pcc(feature, path):
    adj_generate = pearson_adj()
    rlt = []
    for i in tqdm(range(len(feature))):
        graph = feature[i]
        rlt.append(adj_generate.create_adjacency(graph))
    output = open(path['save']+"adjacency_matrices.pkl","wb")
    pickle.dump(rlt,output)
    print("finished adj")



class plv_adj:

    def __init__(self,threshold = None):
        if threshold != None:
            self.threshold = threshold
        else:
            self.threshold = 0

    def create_adjacency(self,graph):
        node_num = graph.shape[0]
        adj_init = np.eye(node_num,node_num)
        for i in range(node_num):
            for j in range(node_num):
                plv = self.get_PLV(graph[i],graph[j])
                if  plv > self.threshold:
                    adj_init[i][j] = plv
        return csr_matrix(adj_init)

    def get_PLV(self,x1,x2):
        """
        get phase-locking value of two 1d-arrays
        """
        sig1_hill=signal.hilbert(x1)
        sig2_hill=signal.hilbert(x2)
        theta1=np.unwrap(np.angle(sig1_hill))
        theta2=np.unwrap(np.angle(sig2_hill))
        complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
        plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
        return plv

def process_adj_plv(feature, path):
    adj_generate = plv_adj()
    rlt = []
    for i in tqdm(range(len(feature))):
        graph = feature[i]
        rlt.append(adj_generate.create_adjacency(graph))
    output = open(path['save']+"adjacency_matrices.pkl","wb")
    pickle.dump(rlt,output)
    print("finished adj")


# if __name__ == "__main__":


#     ReadList = np.load(path['data'], allow_pickle=True)
#     Fold_Num  = ReadList['Fold_len']    # Num of samples of each fold


#     if not os.path.exists(path['save']):
#         os.makedirs(path['save'])

#     preprocess_label()
#     print("Label finished")

#     feature = preprocess_feature()
#     print("feature matrices finished")

#     adj_generator = Adjaency_Generator("knn")
#     adj_generator.get_adj(Fold_Num,feature)


def transform(path, mode="knn"):

    ReadList = np.load(path['data'], allow_pickle=True)
    Fold_Num  = ReadList['Fold_len']    # Num of samples of each fold


    if not os.path.exists(path['save']):
        os.makedirs(path['save'])

    preprocess_label(path)
    print("Label finished")

    feature = preprocess_feature(path)
    print("feature matrices finished")

    adj_generator = Adjaency_Generator(mode)
    adj_generator.get_adj(Fold_Num,feature,path)
