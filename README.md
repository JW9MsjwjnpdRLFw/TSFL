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

- **1. Get Dataset:**
  
  You can download ISRUC-Sleep-S3 dataset by the following command, which will automatically download the raw data and extracted data to `./featureNet/data/ISRUC_S3/`:

  ```shell
  cd ./featureNet
  ./get_ISRUC_S3.sh
  ```

- **2. Data preparation:**

  To facilitate reading, we preprocess the dataset into a single .npz file:

  ```shell
  cd ./featureNet
  python preprocess.py
  ```
  
  In addition, distance based adjacency matrix is provided at `./featureNet/data/ISRUC_S3/DistanceMatrix.npy`.
  
- **3. Configurations of feature extraction module:**

  Write the config file in the format of the example.

  We provide a config file at `./featureNet/config/ISRUC.config`

- **4. Prepare feature extraction module:**

  Run `python train_FeatureNet.py` with -c and -g parameters. After this step, the features learned by a feature net will be stored.

  + -c: The configuration file.
  + -g: The number of the GPU to use. E.g.,`0`,`1,3`. Set this to`-1` if only CPU is used.

  ```shell
  cd ./featureNet
  python train_FeatureNet.py -c ./config/ISRUC.config -g 0
  ```


- **5. TSFL Experiment:**

  Get back to thr Root folder. Run `python fed_experiment.py`.   
  In this command, 3 arguments can be changed.  
  + --model: which GNN model you use in TSFL. Currently supports sage, gat and gcn.
  + --case_name: which connective function you use in TSFL. Currently supports distance, knn, pcc and plv.
  + --data_dir: which dir to save experiment logs, model and temporary files. 

  ```shell
  cd ../
  python fed_experiment.py --model gat --case_name knn --data_dir ./result/ISRUC_S3_knn
  ```

> **Summary of commands to run:**
>
> ```shell
> cd ./featureNet
> ./get_ISRUC_S3.sh
> python preprocess.py
> python train_FeatureNet.py -c ./config/ISRUC.config -g 0
> cd ../
> python fed_experiment.py --model gat --case_name knn --data_dir ./result/ISRUC_S3_knn
> ```
>

