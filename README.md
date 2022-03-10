# TSFL

## Requirements

- Python 3.6
- pytorch 1.9.0
- torchvision 0.9.0
- tensorflow 1.15.0
- Keras 2.3.1
- numpy 1.15.4
- scipy 1.1.0
- scikit-learn 0.19.1
- sklearn 0.19.1
- annoy 1.17.0
- h5py 2.10.0

## How to run

# gnn4spatial_temporarl

## environment
 ```shell
conda create -n gnn python=3.6
bash install_package.sh
  ```


## How to run

- **1. Get Dataset:**
  
  You can download ISRUC-Sleep-S3 dataset by the following url
  http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII
  
  http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels


- **2. Data preparation:**

  To facilitate reading, we preprocess the dataset into a single .npz file:

  ```shell
  python preprocess.py
  ```
  
  In addition, distance based adjacency matrix is provided at `./data/ISRUC_S3/DistanceMatrix.npy`.
  
- **3. Configurations of feature extraction module:**

  Write the config file in the format of the example.

  We provide a config file at `./config/ISRUC.config`

- **4. Prepare feature extraction module:**

  Run `python train_FeatureNet.py` with -c and -g parameters. After this step, the features learned by a feature net will be stored.

  + -c: The configuration file.
  + -g: The number of the GPU to use. E.g.,`0`,`1,3`. Set this to`-1` if only CPU is used.

  ```shell
  python train_FeatureNet.py -c ./config/ISRUC.config -g 0
  ```


- **5. Experiment:**

  Get back to thr Root folder. Run `python gnn_experiment.py`.   
  In this command, 3 arguments can be changed.  
  + --model: which GNN model you use. Currently supports sage, gat and gcn.
  + --case_name: which connective function you use. Currently supports distance, knn, pcc and plv.
  + --data_dir: which dir to save experiment logs, model and temporary files. 

    single machine experiment:

  ```shell
  python gnn_experiment.py --model gat --case_name knn --data_dir ./result/ISRUC_S3_knn/
  python gnn_experiment.py --model gat --case_name plv --data_dir ./result/ISRUC_S3_plv/
  python gnn_experiment.py --model gat --case_name distance --data_dir ./result/ISRUC_S3_distance/
  python gnn_experiment.py --model gat --case_name pcc --data_dir ./result/ISRUC_S3_distance/
  ```
    multiple machines experiment:
    
  ```shell
  python fed_experiment.py --model gat --case_name knn --data_dir ./result/ISRUC_S3_knn  
  python fed_experiment.py --model gat --case_name plv --data_dir ./result/ISRUC_S3_plv  
  python fed_experiment.py --model gat --case_name distance --data_dir ./result/ISRUC_S3_distance  
  python fed_experiment.py --model gat --case_name pcc --data_dir ./result/ISRUC_S3_pcc  

  ```


> **Summary of commands to run:**
>
> ```shell
> ./get_ISRUC_S3.sh
> python preprocess.py
> python train_FeatureNet.py -c ./config/ISRUC.config -g 0
> python fed_experiment.py --model gat --case_name knn --data_dir ./result/ISRUC_S3_knn
> ```
>

