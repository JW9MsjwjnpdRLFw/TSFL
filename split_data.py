import argparse
import os
import numpy as np
import pickle

# from utils import *

# from DataTransformer import transform



def run(path,case_name):
    path = path[:-1]

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





    np.random.seed(0)


    # path = "./result/ISRUC_S3_pcc"
    # case_name = "pcc"

    transformed_path = path+"/"+case_name
    if not os.path.exists(transformed_path):
        adj_matrices, feature_matrices, labels = get_data(path)
        os.mkdir(transformed_path)
        numb_participants = 5

        train_mask, test_mask = niid_split_mask(labels)

        os.mkdir(transformed_path+"/test")

        writer=open("{}/test/{}.pkl".format(transformed_path, "adjacency_matrices"),'wb')
        pickle.dump(np.array(adj_matrices)[test_mask], writer)
        writer.close()

        writer=open("{}/test/{}.pkl".format(transformed_path, "feature_matrices"),'wb')
        pickle.dump(np.array(feature_matrices)[test_mask], writer)
        writer.close()

        np.save("{}/test/{}.npy".format(transformed_path, "labels"), labels[test_mask])

        for i, mask in enumerate(train_mask):
            os.mkdir(transformed_path+"/Agent{}".format(i))
            writer=open("{}/Agent{}/{}.pkl".format(transformed_path, i, "adjacency_matrices"),'wb')
            pickle.dump(np.array(adj_matrices)[mask], writer)
            writer.close()

            writer=open("{}/Agent{}/{}.pkl".format(transformed_path, i, "feature_matrices"),'wb')
            pickle.dump(np.array(feature_matrices)[mask], writer)
            writer.close()

            np.save("{}/Agent{}/{}.npy".format(transformed_path, i, "labels"), labels[mask])