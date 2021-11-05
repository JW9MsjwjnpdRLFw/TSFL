import argparse
import os
import numpy as np
import pickle

from sklearn import metrics

import torch.utils.data
import torch.nn as nn
import torch.utils.data as data

from GCN_model.model import GAT, GCN, Graphsage 
from GCN_model.utlis.datasets import MoleculesDataset
from GCN_model.utlis.utils import *

def random_split_mask(total_length, mask_numb):
    index = np.arange(total_length)
    mask_len = int(total_length/mask_numb)
    np.random.shuffle(index)
    masks = []
    for i in range(mask_numb-1):
        mask = index[mask_len*i:mask_len*(i+1)]
        masks.append(mask)
    masks.append(index[mask_len*(mask_numb-1):])
    return masks

def niid_split_mask(labels):
    index = np.arange(labels.shape[0])
    mask_len = int(labels.shape[0]/4)
    np.random.shuffle(index)
    test_mask = index[:mask_len]

    label_mask = [[] for i in range(labels.shape[1])]
    for i, label in enumerate(labels.argmax(axis=1)):
        if i not in test_mask:
            label_mask[label].append(i)

    train_mask = [
        label_mask[0][:int(len(label_mask[0])*1/3)]+label_mask[1][:int(len(label_mask[1])*1/3)]+label_mask[2][:int(len(label_mask[2])*1/3)],
        label_mask[1][int(len(label_mask[1])*1/3):int(len(label_mask[1])*2/3)]+label_mask[2][int(len(label_mask[2])*1/3):int(len(label_mask[2])*2/3)]+label_mask[3][:int(len(label_mask[3])*1/3)],
        label_mask[2][int(len(label_mask[2])*2/3):]+label_mask[3][int(len(label_mask[3])*1/3):int(len(label_mask[3])*2/3)]+label_mask[4][:int(len(label_mask[4])*1/3)],
        label_mask[3][int(len(label_mask[3])*2/3):]+label_mask[4][int(len(label_mask[4])*1/3):int(len(label_mask[4])*2/3)]+label_mask[0][int(len(label_mask[0])*1/3):int(len(label_mask[0])*2/3)],
        label_mask[4][int(len(label_mask[4])*2/3):]+label_mask[0][int(len(label_mask[0])*2/3):]+label_mask[1][int(len(label_mask[1])*2/3):],
    ]
    print("Train mask")
    print(sum(labels[np.array(train_mask[0])]), len(labels[np.array(train_mask[0])]))
    print(sum(labels[np.array(train_mask[1])]), len(labels[np.array(train_mask[1])]))
    print(sum(labels[np.array(train_mask[2])]), len(labels[np.array(train_mask[2])]))
    print(sum(labels[np.array(train_mask[3])]), len(labels[np.array(train_mask[3])]))
    print(sum(labels[np.array(train_mask[4])]), len(labels[np.array(train_mask[4])]))
    
    print("Test mask")
    print(sum(labels[test_mask]), len(labels[test_mask]))
    
    return train_mask, test_mask


def get_data(path):
    with open(path + '/adjacency_matrices.pkl', 'rb') as f:
        adj_matrices = pickle.load(f)

    with open(path + '/feature_matrices.pkl', 'rb') as f:
        feature_matrices = pickle.load(f)

    labels = np.load(path + '/labels.npy')

    return adj_matrices, feature_matrices, labels


def get_dataloader(path, compact=True, normalize_features=False, normalize_adj=False):
    train_adj_matrices, train_feature_matrices, train_labels = get_data(path)
    
    train_dataset = MoleculesDataset(train_adj_matrices, train_feature_matrices, train_labels, path, compact=compact, split='train')
   
    collator = WalkForestCollator(normalize_features=normalize_features) if compact \
        else DefaultCollator(normalize_features=normalize_features, normalize_adj=normalize_adj)

    # IT IS VERY IMPORTANT THAT THE BATCH SIZE = 1. EACH BATCH IS AN ENTIRE MOLECULE.
    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator, pin_memory=True)
 
    return train_dataloader


def sync_participants(participant, global_m):
    participant.load_state_dict(global_m.state_dict())
    

def weight_aggregate(global_model, participants):
    """
    This function has aggregation method 'mean'
    将全部子节点mobilenet模型的更新，同步到全局模型。
    上行和下行被部署在这个函数里面。
    更新方式为全部子节点权重取平均。
    Prune也在这部分进行。
    Prune：剪枝，把符合条件的权重置0

    Parameters:
    global_model    - 全局模型
    participants    - 全部子节点mobilenet模型

    Returns:
        无返回
    """

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([participants[i].state_dict()[k].float() for i in range(len(participants))],0).mean(0)
    global_model.load_state_dict(global_dict)

    for model in participants:
        model.load_state_dict(global_model.state_dict())

def get_model(args, feat_dim, num_cats):
    if args.model == 'gcn':
        model = GCN.GCNClassification(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                               args.readout_hidden_dim, args.graph_embedding_dim, num_cats,
                               sparse_adj=args.sparse_adjacency)
    elif args.model == 'gat':
        model = GAT.GatClassification(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                               args.alpha, args.num_heads, args.readout_hidden_dim, args.graph_embedding_dim, num_cats)
    elif args.model == 'graphsage':
        model = Graphsage.GraphSageClassification(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                                args.readout_hidden_dim, args.graph_embedding_dim, num_cats)
    else:
        raise Exception('No such model')
    
    return model


def PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:", file=saveFile)
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['Wake','N1','N2','N3','REM'],
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return


def calculate_loss(model, dataloader, batch_size, device, criterion, is_sage):

    batch_loss = 0
    for idx in range(batch_size):
        if is_sage:
            forest, feature_matrix, label, mask = next(dataloader)
            if torch.all(mask == 0).item():
                continue
            
            forest = [level.to(device=device, dtype=torch.long, non_blocking=True) for level in forest]
            feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
            label = label.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
            
            logits = model(forest, feature_matrix)

        else:
            adj_matrix, feature_matrix, label, mask = next(dataloader)
            if torch.all(mask == 0).item():
                continue

            adj_matrix = adj_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
            feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
            label = label.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
            
            logits = model(adj_matrix, feature_matrix)
        loss = criterion(logits, label) * mask
        loss = loss.sum() / mask.sum()

        batch_loss += loss
    return batch_loss/batch_size 



def acc_f1(model, test_loader, device, is_sage):
    model.eval()
    model.to(device)

    with torch.no_grad():
        y_pred = []
        y_true = []
        
        if is_sage:
            for mol_idx, (forest, feature_matrix, label, mask) in enumerate(test_loader):
                forest = [level.to(device=device, dtype=torch.long, non_blocking=True) for level in forest]
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(forest, feature_matrix)

                y_pred.append(nn.Sigmoid()(logits).cpu().numpy())
                y_true.append(label.numpy())
        else:
            for mol_idx, (adj_matrix, feature_matrix, label, mask) in enumerate(test_loader):
                adj_matrix = adj_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(adj_matrix, feature_matrix)

                y_pred.append(nn.Sigmoid()(logits).cpu().numpy())
                y_true.append(label.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)

    return metrics.accuracy_score(true, pred), metrics.f1_score(true, pred, average='macro'), metrics.confusion_matrix(true,pred)



