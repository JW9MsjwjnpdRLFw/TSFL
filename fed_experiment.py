import argparse
import os
import numpy as np
import pickle

import torch.utils.data
import torch.nn as nn

from GCN_model.utlis.utils import *

from utils import *

from DataTransformer import transform


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument('--case_name', type=str, default='knn',
                        help='connective functions ("distance","knn","pcc","plv")')

    parser.add_argument('--data_dir', type=str, default="./result/ISRUC_S3_knn", help='Data directory')

    parser.add_argument('--model', type=str, default='graphsage',
                        help='Model name. Currently supports SAGE, GAT and GCN.')

    parser.add_argument('--normalize_features', type=bool, default=False,
                        help='Whether or not to symmetrically normalize feat matrices')

    parser.add_argument('--normalize_adjacency', type=bool, default=False,
                        help='Whether or not to symmetrically normalize adj matrices')

    parser.add_argument('--sparse_adjacency', type=bool, default=False,
                        help='Whether or not the adj matrix is to be processed as a sparse matrix')

    parser.add_argument('--hidden_size', type=int, default=32, help='Size of GNN hidden layer')

    parser.add_argument('--node_embedding_dim', type=int, default=32,
                        help='Dimensionality of the vector space the atoms will be embedded in')

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for LeakyRelu used in GAT')

    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads used in GAT')

    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout used between GraphSAGE layers')

    parser.add_argument('--readout_hidden_dim', type=int, default=64, help='Size of the readout hidden layer')

    parser.add_argument('--graph_embedding_dim', type=int, default=64,
                        help='Dimensionality of the vector space the molecule will be embedded in')

    parser.add_argument('--client_optimizer', type=str, default='adam', metavar="O",
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0015, metavar='LR',
                        help='learning rate (default: 0.0015)')

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS',
                        help='batch size (default: batch_size)')

    parser.add_argument('--wd', help='weight decay parameter;', metavar="WD", type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--frequency_of_the_test', type=int, default=5, help='How frequently to run eval')

    parser.add_argument('--device', type=str, default="cuda:0", metavar="DV", help='gpu device for training')

    parser.add_argument('--metric', type=str, default='roc-auc',
                        help='Metric to be used to evaluate classification models')

    parser.add_argument('--test_freq', type=int, default=1024, help='How often to test')

    args = parser.parse_args()

    return args


def train_model(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    path = args.data_dir
    case_name = args.case_name

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size

    transformed_path = path + "/" + case_name
    if not os.path.exists(transformed_path):
        adj_matrices, feature_matrices, labels = get_data(path)
        os.mkdir(transformed_path)
        numb_participants = 5

        train_mask, test_mask = niid_split_mask(labels)

        os.mkdir(transformed_path + "/test")

        writer = open("{}/test/{}.pkl".format(transformed_path, "adjacency_matrices"), 'wb')
        pickle.dump(np.array(adj_matrices)[test_mask], writer)
        writer.close()

        writer = open("{}/test/{}.pkl".format(transformed_path, "feature_matrices"), 'wb')
        pickle.dump(np.array(feature_matrices)[test_mask], writer)
        writer.close()

        np.save("{}/test/{}.npy".format(transformed_path, "labels"), labels[test_mask])

        for i, mask in enumerate(train_mask):
            os.mkdir(transformed_path + "/Agent{}".format(i))
            writer = open("{}/Agent{}/{}.pkl".format(transformed_path, i, "adjacency_matrices"), 'wb')
            pickle.dump(np.array(adj_matrices)[mask], writer)
            writer.close()

            writer = open("{}/Agent{}/{}.pkl".format(transformed_path, i, "feature_matrices"), 'wb')
            pickle.dump(np.array(feature_matrices)[mask], writer)
            writer.close()

            np.save("{}/Agent{}/{}.npy".format(transformed_path, i, "labels"), labels[mask])

    compact = (args.model == 'graphsage')

    # 加载数据
    train_data_set = []
    test_data_set = []

    feat_dim = 256
    num_cats = 5

    num_participants = len(os.listdir(transformed_path)) - 1  # 分布式网络中，子节点的数量

    for folder in os.listdir(transformed_path):
        print("Load: {}/{}".format(transformed_path, folder))

        loaded_data = get_dataloader(transformed_path + "/" + folder,
                                     compact=compact,
                                     normalize_features=False,
                                     normalize_adj=False)
        _, feature_matrices, labels = get_data(transformed_path + "/" + folder)
        feat_dim = feature_matrices[0].shape[1]
        num_cats = labels[0].shape[0]

        print("lenth    = %d" % len(loaded_data))
        print("feat_dim = %d" % feat_dim)
        print("num_cats = %d" % num_cats)
        print()

        if folder != "test":
            train_data_set.append(loaded_data)
            print("Train mask")
            print(sum(labels))
        else:
            test_data_set.append(loaded_data)
            print("Test mask")
            print(sum(labels))

            # 初始化模型
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device == 'cuda:0') else "cpu")

    os.mkdir(path + "/" + args.model)
    logFile = open(path + "/" + args.model + "/log.txt", 'a+')
    print("logfile:", path + "/" + args.model + "/log.txt")

    global_model = get_model(args, feat_dim, num_cats)
    print(global_model.readout)
    print(global_model.readout, file=logFile)

    participants = [get_model(args, feat_dim, num_cats) for i in range(num_participants)]
    print("len_participants:", len(participants))

    for participant_model in participants:
        sync_participants(participant_model, global_model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model in participants:
        model.to(device=device, dtype=torch.float32, non_blocking=True)
        model.train()

    global_model.to(device=device, dtype=torch.float32, non_blocking=True)
    global_model.train()

    train_loader = train_data_set
    test_loader = test_data_set

    history_train = [[] for i in participants]
    history_test = []
    history_CM = []

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    opts = [torch.optim.Adam(model.parameters(), lr=lr) for model in participants]

    best_model = None
    best_f1 = 0

    for e in range(epochs):
        for mol_idxs in range(int(len(train_loader[0]) / batch_size)):
            participants_loss_train = []

            for i in range(len(participants)):
                batch_loss = calculate_loss(model=participants[i],
                                            dataloader=iter(train_loader[i]),
                                            batch_size=batch_size,
                                            device=device,
                                            criterion=criterion,
                                            is_sage=compact)

                optimizer = opts[i]
                optimizer.zero_grad()

                participants_loss_train.append(batch_loss)
                batch_loss.backward()
                optimizer.step()

                history_train[i].append(batch_loss)
            weight_aggregate(global_model, participants)

            if mol_idxs % 5 == 0 or mol_idxs == int(len(train_loader[0]) / batch_size) - 1:
                global_loss_test = calculate_loss(model=global_model,
                                                  dataloader=iter(test_loader[0]),
                                                  batch_size=batch_size * 8,
                                                  device=device,
                                                  criterion=criterion,
                                                  is_sage=compact)
                acc, f1, cm = acc_f1(global_model, iter(test_loader[0]), device, compact)
                print(
                    'Train epoch {:^3} at batch {:^5} with global accuracy {:5.4f}, F1 score {:5.4f}, test loss {:5.4f}, participants train loss [{:5.4f}, {:5.4f}, {:5.4f}] [({:2.0f}%)]'.format(
                        e, mol_idxs,
                        acc, f1,
                        global_loss_test,
                        participants_loss_train[0], participants_loss_train[1], participants_loss_train[2],
                        mol_idxs / int(len(train_loader[0]) / batch_size) * 100), file=logFile)
                history_test.append(global_loss_test)
                print(cm, file=logFile)

                print(
                    'Train epoch {:^3} at batch {:^5} with global accuracy {:5.4f}, F1 score {:5.4f}, test loss {:5.4f}, participants train loss [{:5.4f}, {:5.4f}, {:5.4f}] [({:2.0f}%)]'.format(
                        e, mol_idxs,
                        acc, f1,
                        global_loss_test,
                        participants_loss_train[0], participants_loss_train[1], participants_loss_train[2],
                        mol_idxs / int(len(train_loader[0]) / batch_size) * 100))
                history_test.append(global_loss_test)
                # print(cm)
                history_CM.append(cm)

                if f1 > best_f1:
                    best_model = global_model
                    best_f1 = f1
        print("", file=logFile)
        print()
    logFile.close()

    np.save(path + "/" + args.model + "/history_CM", np.array(history_CM))
    np.save(path + "/" + args.model + "/history_test", np.array([loss.cpu().detach().numpy() for loss in history_test]))
    np.save(path + "/" + args.model + "/history_train",
            np.array([[loss.cpu().detach().numpy() for loss in parti] for parti in history_train]))

    torch.save(global_model, path + "/" + args.model + "/global_model.model")
    for i, participant in enumerate(participants):
        torch.save(participant, path + "/" + args.model + "/participant_model_{}.model".format(i))

    best_model.eval()
    best_model.to(device)

    with torch.no_grad():
        y_pred = []
        y_true = []
        masks = []

        if compact:
            for mol_idx, (forest, feature_matrix, label, mask) in enumerate(iter(test_loader[0])):
                forest = [level.to(device=device, dtype=torch.long, non_blocking=True) for level in forest]
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = best_model(forest, feature_matrix)

                y_pred.append(nn.Sigmoid()(logits).cpu().numpy())
                y_true.append(label.numpy())
                masks.append(mask.numpy())
        else:
            for mol_idx, (adj_matrix, feature_matrix, label, mask) in enumerate(iter(test_loader[0])):
                adj_matrix = adj_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = best_model(adj_matrix, feature_matrix)

                y_pred.append(nn.Sigmoid()(logits).cpu().numpy())
                y_true.append(label.numpy())
                masks.append(mask.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    masks = np.array(masks)

    AllPred = np.argmax(y_pred, axis=1)
    AllTrue = np.argmax(y_true, axis=1)
    PrintScore(AllTrue, AllPred, savePath=path + "/" + args.model + "/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    path = {
        'data': "./data/ISRUC_S3/ISRUC_S3.npz",
        'save': args.data_dir + "/",
        "cheb_k": 3,
        "disM": "./data/ISRUC_S3/DistanceMatrix.npy",
        "feature": './output/Feature_0.npz'
    }
    transform(path, args.case_name)

    train_model(args)

    # python fed_experiment.py --model gat --case_name knn --data_dir ./result/ISRUC_S3_knn
    # python fed_experiment.py --model gcn --case_name knn --data_dir ./result/ISRUC_S3_knn
    # python fed_experiment.py --model graphsage --case_name knn --data_dir ./result/ISRUC_S3_knn