import configparser
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.sparse.linalg import eigs
import keras


from sklearn.utils import check_array
from sklearn.neighbors import NearestNeighbors, LSHForest
from sklearn.utils.validation import check_random_state
from annoy import AnnoyIndex


from scipy.sparse import (csr_matrix, spdiags, eye)
from sklearn.utils.graph import graph_laplacian
##########################################################################################
# Read configuration file ################################################################

def ReadConfig(configfile):
    config = configparser.ConfigParser()
    print('Config: ', configfile)
    config.read(configfile)
    cfgPath = config['path']
    cfgFeat = config['feature']
    cfgTrain = config['train']
    cfgModel = config['model']
    return cfgPath, cfgFeat, cfgTrain, cfgModel

##########################################################################################
# Add context to the origin data and label ###############################################

def AddContext_MultiSub(x, y, Fold_Num, context, i):
    '''
    input:
        x       : [N,V,F];
        y       : [N,C]; (C:num_of_classes)
        Fold_Num: [kfold];
        context : int;
        i       : int (i-th fold)
    return:
        x with contexts. [N',V,F]
    '''
    cut = context // 2
    fold = Fold_Num.copy()
    fold = np.delete(fold, -1)
    id_del = np.concatenate([np.cumsum(fold) - i for i in range(1, context)])
    id_del = np.sort(id_del)

    x_c = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
    for j in range(cut, x.shape[0] - cut):
        x_c[j - cut] = x[j - cut:j + cut + 1]

    x_c = np.delete(x_c, id_del, axis=0)
    y_c = np.delete(y[cut: -cut], id_del, axis=0)
    return x_c, y_c

def AddContext_SingleSub(x, y, context):
    cut = int(context / 2)
    x_c = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
    for i in range(cut, x.shape[0] - cut):
        x_c[i - cut] = x[i - cut:i + cut + 1]
    y_c = y[cut:-cut]
    return x_c, y_c

##########################################################################################
# Instantiation operation ################################################################

def Instantiation_optim(name, lr):
    if   name=="adam":
        opt = keras.optimizers.Adam(lr=lr)
    elif name=="RMSprop":
        opt = keras.optimizers.RMSprop(lr=lr)
    elif name=="SGD":
        opt = keras.optimizers.SGD(lr=lr)
    else:
        assert False,'Config: check optimizer, may be not implemented.'
    return opt

def Instantiation_regularizer(l1, l2):
    if   l1!=0 and l2!=0:
        regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
    elif l1!=0 and l2==0:
        regularizer = keras.regularizers.l1(l1)
    elif l1==0 and l2!=0:
        regularizer = keras.regularizers.l2(l2)
    else:
        regularizer = None
    return regularizer

##########################################################################################
# Print score between Ytrue and Ypred ####################################################

def PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
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

##########################################################################################
# Print confusion matrix and save ########################################################

def ConfusionMatrix(y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n=cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]*100,'.2f')+'%\n'+format(cm_n[i, j],'d'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath+title+".png")
    plt.show()
    return ax

##########################################################################################
# Draw ACC / loss curve and save #########################################################

def VariationCurve(fit,val,yLabel,savePath,figsize=(9, 6)):
    plt.figure(figsize=figsize)
    plt.plot(range(1,len(fit)+1), fit,label='Train')
    plt.plot(range(1,len(val)+1), val, label='Val')
    plt.title('Model ' + yLabel)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(savePath + 'Model_' + yLabel + '.png')
    plt.show()
    return

# compute \tilde{L}

def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    ----------
    Parameters
    W: np.ndarray, shape is (N, N), N is the num of vertices
    ----------
    Returns
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis = 1))
    L = D - W
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

##########################################################################################
# compute a list of chebyshev polynomials from T_0 to T_{K-1} ############################

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    ----------
    Parameters
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    ----------
    Returns
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = np.array([np.identity(N), L_tilde.copy()])
    for i in range(2, K):
        cheb_polynomials = np.append(
            cheb_polynomials,
            [2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]],
            axis=0)
    return cheb_polynomials


# TODO: reference all of the nearest neighbor libraries used


class KnnSolver(object):
    """KnnSolver class implements some nearest neighbor algorithms

    Parameters
    ----------
    n_neighbors : int, default = 2
        number of nearest neighbors

    radius : int, default = 1
        length of the radius for the neighbors in distance

    algorithm : str, default = 'annoy'
        ['auto'|'annoy'|'brute'|'kd_tree'|'ball_tree'|'pyflann'|'cyflann']
        algorithm to find the k-nearest or radius-nearest neighbors

    algorithm_kwargs : dict, default = None
        a dictionary of key word values for specific arguments on each algorithm

    References
    ----------
    * sklearn: brute, kd_tree, ball_tree
        https://goo.gl/2noI11
    * annoy
        https://github.com/spotify/annoy
    * nmslib (TODO)
    * pyflann (TODO)
        https://github.com/primetang/pyflann
    * cyflann (TODO)
        https://github.com/dougalsutherland/cyflann

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 5th February, 2017
    Email   : emanjohnson91@gmail.com
    """
    def __init__(self, n_neighbors=2, radius=1.5, method='knn', algorithm='brute',
                 random_state=None, algorithm_kwargs=None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.method = method
        self.algorithm = algorithm
        self.algorithm_kwargs = algorithm_kwargs
        self.random_state = random_state

    def find_knn(self, data):

        # check random state
        self.random_state = check_random_state(self.random_state)

        # check the array
        data = check_array(data)

        # TODO: check kwargs
        self.check_nn_solver_()

        # sklearn (auto, brute, kd_tree, ball_tree)
        if self.algorithm in ['auto', 'brute', 'kd_tree', 'ball_tree']:

            # initialize nearest neighbors model
            nbrs_model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                          radius=self.radius,
                                          algorithm=self.algorithm,
                                          **self.algorithm_kwargs)

            # fit the model to the data
            nbrs_model.fit(data)

            # extract distances and indices
            if self.method in ['knn']:
                distances, indices = \
                    nbrs_model.kneighbors(data, n_neighbors=self.n_neighbors,
                                          return_distance=True)

            elif self.method in ['radius']:
                distances, indices = \
                    nbrs_model.radius_neighbors(data, radius=self.radius,
                                                return_distance=True)

            else:
                raise ValueError('Unrecognized connectivity method.')

            return distances, indices

        elif self.algorithm in ['annoy']:

            if self.algorithm_kwargs is None:
                return ann_annoy(data, n_neighbors=self.n_neighbors)
            else:
                return ann_annoy(data, n_neighbors=self.n_neighbors,
                                 **self.algorithm_kwargs)

        elif self.algorithm in ['pyflann']:
            # TODO: implement pyflann nn method
            raise NotImplementedError('Method has not been completed yet.')

        elif self.algorithm in ['cyflann']:
            # TODO: implement cyflann nn method
            raise NotImplementedError('cyflann has not been completed yet.')

        else:
            raise ValueError('Unrecognized algorithm.')

    def check_nn_solver_(self):

        # check for None type
        if self.algorithm_kwargs is None:
            self.algorithm_kwargs = {}

        # TODO: check sklearn-brute
        # TODO: check sklearn-kd_tree
        # TODO: check sklearn-ball_tree
        # TODO: check sklearn-lshf
        # TODO: check annoy
        # TODO: check pyflann
        # TODO: check cyflann

        return self


def ann_annoy(data, n_neighbors=2, metric='euclidean', trees=10):
    """My approximate nearest neighbor function that uses the ANNOY python
    package

    Parameters
    ----------
    data : array, (N x D)

    n_neighbors : int, default = 2

    metric : str, default = 'euclidean'

    trees : int, default = 10

    Returns
    -------
    distances : array, (N x n_neighbors)

    indices : array, (N x n_neighbors)

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    :
    Email   : emanjohnson91@gmail.com
    """
    datapoints = data.shape[0]
    dimension = data.shape[1]

    # initialize the annoy database
    ann = AnnoyIndex(dimension, metric=metric)

    # store the datapoints
    for (i, row) in enumerate(data):
        ann.add_item(i, row.tolist())

    # build the index
    ann.build(trees)

    # find the k-nearest neighbors for all points
    indices = np.zeros((datapoints, n_neighbors), dtype='int')
    distances = indices.copy().astype(np.float)

    # extract the distance values
    for i in range(0, datapoints):
        indices[i, :] = ann.get_nns_by_item(i, n_neighbors)

        for j in range(0, n_neighbors):
            distances[i, j] = ann.get_distance(i, indices[i, j])

    return distances, indices


# import standard scientific packages


class Adjacency(object):
    """
    A class to construct the adjacency matrix.

    Parameters
    ----------
    X : array, ( N x D )
        data matrix

    n_neighbors : int, (default = 5)
        number of nearest neighbors

    radius : int, (default = 1.5)
        radius to find the nearest neighbors

    method : str, (default = 'radius')
        ['radius' | 'knn' ]
        the type of nearest neighbors algorithm

    weight : str, (default = 'heat')
        ['connectivity'|'heat'|'angle']
        the scaling kernel function to use on the distance values found from
        the neighbors algorithm

    algorithm : str, (default = 'brute')
        ['annoy'|'brute'|'kd_tree'|'ball_tree'|'hdidx'|'pyflann'|'cyflann']
        algorithm to find the k-nearest or radius-nearest neighbors
    """

    def __init__(self, n_neighbors=10, radius=1.5, algorithm='brute',
                 mode='distance', metric='euclidean', method='knn', weight='heat',
                 gamma=1.0,
                 algorithm_kwargs=None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.mode = mode
        self.metric = metric
        self.method = method
        self.weight = weight
        self.gamma = gamma
        self.algorithm_kwargs = algorithm_kwargs

    def create_adjacency(self, X):

        # Check array
        X = check_array(X)

        self.n_samples = X.shape[0]

        # Initiate KNN Class
        knn_model = KnnSolver(n_neighbors=self.n_neighbors,
                              radius=self.radius,
                              algorithm=self.algorithm,
                              method=self.method,
                              algorithm_kwargs=self.algorithm_kwargs)

        # Find the nearest neighbors distances
        distances, indices = knn_model.find_knn(X)

        # construct adjacency
        self.adjacency_matrix = self.adjacency(distances, indices)

        return self.adjacency_matrix

    def adjacency(self, distances, indices):

        # Construct CSR Matrix representation of NN graph
        if self.method in ['radius']:
            # construct CSR matrix representation of the NN graph
            if self.mode in ['connectivity']:
                A_ind = indices
                A_data = None

            elif self.mode not in ['distance']:
                A_ind = indices
                A_data = np.concatenate(list(distances))

            else:
                raise ValueError(
                    'Unsupported mode, must be one of "connectivity", '
                    'or "distance" but got {s} instead'.format(self.mode))

            self.n_datapoints = A_ind.shape[0]
            n_neighbors = np.array([len(a) for a in A_ind])
            A_ind = np.concatenate(list(A_ind))

            if A_data is None:
                A_data = np.ones(len(A_ind))
            A_indptr = np.concatenate((np.zeros(1, dtype=int),
                                       np.cumsum(n_neighbors)))

        elif self.method in ['knn']:

            self.n_datapoints = self.n_samples

            n_neighbors = distances.shape[1]

            n_nonzero = self.n_samples * n_neighbors
            A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

            # construct CSR matrix representation of the k-NN graph
            if self.weight in ['connectivity']:
                A_data = np.ones(self.n_samples * n_neighbors)
                A_ind = indices

            elif self.weight not in ['distance']:
                A_data, A_ind = distances, indices
                A_data = np.ravel(A_data)

            else:
                raise ValueError(
                    'Unsupported mode, must be one of "connectivity", '
                    'or "distance" but got {s} instead'.format(self.mode))

        else:
            raise ValueError(
                'Unrecognized method of graph construction. Must be '
                '"knn" or "radius" but got "{alg}" instead'.format(alg=self.method))

        # Compute the weights
        A_data = self.compute_weights(A_data)

        # Create Sparse Matrix
        adjacency_mat = self._create_sparse_matrix(A_data, A_ind, A_indptr)

        return adjacency_mat

    def compute_weights(self, data):

        if self.weight in ['connectivity']:
            pass

        elif self.weight in ['heat']:
            data = np.exp(-data ** 2 / self.gamma ** 2)

        elif self.weight in ['angle']:
            data = np.exp(-np.arccos(1 - data))

        else:
            raise ValueError('Sorry. Unrecognized affinity weight.')

        return data

    def _create_sparse_matrix(self, data, ind, indptr):

        if self.method in ['knn']:
            adjacency_mat = csr_matrix(
                (data, ind.ravel(), indptr),
                shape=(self.n_datapoints, self.n_samples)
            )


        elif self.method in ['radius']:
            adjacency_mat = csr_matrix(
                (data, ind, indptr),
                shape=(self.n_datapoints, self.n_samples))

        # make sure the matrix is symmetric
        adjacency_mat = maximum(adjacency_mat, adjacency_mat.T)

        return adjacency_mat


# compute adjacency matrix
def adjacency(X, n_neighbors=10, radius=1.5, algorithm='brute', method='radius',
              weight='connectivity', nearest_neighbors_kwargs=None,
              adjacency_kwargs=None):
    """Computes an adjacency matrix

    Parameters
    ----------
    X : array, ( N x D )
        data matrix

    n_neighbors : int, default = 5
        number of nearest neighbors

    radius : int, default = 1.5
        radius to find the nearest neighbors

    method : str, default = 'radius'
        ['radius' | 'knn' ]
        the type of nearest neighbors algorithm

    weight : str, default = 'connectivity'
        ['connectivity'|'heat'|'angle']
        the scaling kernel function to use on the distance values found from
        the neighbors algorithm

    algorithm : str, default = 'brute'
        ['annoy'|'brute'|'kd_tree'|'ball_tree'|'hdidx'|'pyflann'|'cyflann']
        algorithm to find the k-nearest or radius-nearest neighbors

    nearest_neighbors_kwargs : dict, default = None
        a dictionary of key values for the KnnSolver class

    adjacency_kwargs : dict, default = None
        a dictionary of key values for the adjacency matrix construction

    Returns
    -------
    adjacency_matrix : array, ( N x N )
        a weighted adjacency matrix

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 5th February, 2017
    Email   : emanjohnson91@gmail.com

    References
    ----------
    TODO: code references
    TODO: paper references
    """

    if algorithm in ['annoy'] and method in ['radius']:
        method = 'knn'

    # initialize the knn model with available parameters
    knn_model = KnnSolver(n_neighbors=n_neighbors,
                          radius=radius,
                          algorithm=algorithm,
                          method=method,
                          algorithm_kwargs=nearest_neighbors_kwargs)

    # find the nearest neighbors indices and distances
    distances, indices = knn_model.find_knn(X)

    # construct adjacency matrix
    if adjacency_kwargs is None:
        adjacency_kwargs = {}

    adjacency_matrix = create_adjacency(distances, indices, X, method=method,
                                        weight=weight,
                                        **adjacency_kwargs)

    return adjacency_matrix


def create_adjacency(distances, indices, data, method='radius', mode='distance',
                     weight='connectivity', gamma=1.0):
    """This function will create a sparse symmetric
    weighted adjacency matrix from nearest neighbors
    and their corresponding distances.

    Parameters:
    -----------
    indices : array, (N x k)
        an Nxk array where M are the number of data points and k
        are the k-1 nearest neighbors connected to that data point M.

    distances : array, (N x k)
        an MxN array where N are the number of data points and k are
        the k-1 nearest neighbor distances connected to that data point M.

    data : array, (N x D)
        an NxD array where N is the number of data points and D is the number
        of dimensions.

    method : str, default = 'knn'
        ['knn' | 'radius']
        base algorithm to find the nearest neighbors

    weight : str, default = 'heat'
        ['heat' | 'angle' | 'connectivity']
        weights to put on the data points

    gamma : float, default 1.0
        the spread for the weight function

    Returns:
    --------
    Adjacency Matrix          - a sparse MxM sparse weighted adjacency
                                matrix.

    References
    ----------
    Uses code from the sklearn library, specifically the neighbors, base
    class with functions k_neighbors_graph and radius_neighbors_graph
        https://goo.gl/DKtpBX
    """
    # Separate, tile and ravel the neighbours from their
    # corresponding points

    # check array (sparse is allowed)
    data = check_array(data, accept_sparse=['csr', 'csc', 'coo'])

    # dimensions for samples
    n_samples2 = data.shape[0]

    if method in ['radius']:

        # construct CSR matrix representation of the NN graph
        if weight in ['connectivity']:
            A_ind = indices
            A_data = None

        elif weight not in ['distance']:
            A_ind = indices
            A_data = np.concatenate(list(distances))

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                'or "distance" but got %s instead' % mode)

        n_samples1 = A_ind.shape[0]
        n_neighbors = np.array([len(a) for a in A_ind])
        A_ind = np.concatenate(list(A_ind))
        if A_data is None:
            A_data = np.ones(len(A_ind))
        A_indptr = np.concatenate((np.zeros(1, dtype=int),
                                   np.cumsum(n_neighbors)))

    elif method in ['knn']:

        n_samples1 = data.shape[0]

        n_neighbors = distances.shape[1]

        n_nonzero = n_samples1 * n_neighbors
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

        # construct CSR matrix representation of the k-NN graph
        if weight in ['connectivity']:
            A_data = np.ones(n_samples1 * n_neighbors)
            A_ind = indices

        elif weight not in ['distance']:
            A_data, A_ind = distances, indices
            A_data = np.ravel(A_data)

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity" '
                'or "distance" but got "%s" instead' % mode)

    else:
        raise ValueError(
            'Unrecognized method of graph construction. Must be '
            '"knn" or "radius" but got "{alg}" instead'.format(alg=method))

    # compute weights
    if weight in ['connectivity']:
        pass

    elif weight in ['heat']:
        A_data = np.exp(-A_data ** 2 / gamma ** 2)

    elif weight in ['angle']:
        A_data = np.exp(-np.arccos(1 - A_data))

    else:
        raise ValueError('Sorry. Unrecognized affinity weight.')

    # Create the sparse matrix
    if method in ['knn']:
        adjacency_mat = csr_matrix((A_data, A_ind.ravel(), A_indptr),
                                   shape=(n_samples1, n_samples2))

    elif method in ['radius']:
        adjacency_mat = csr_matrix((A_data, A_ind, A_indptr),
                                   shape=(n_samples1, n_samples2))

    # Make sure the matrix is symmetric
    adjacency_mat = maximum(adjacency_mat, adjacency_mat.T)

    return adjacency_mat


def create_constraint(adjacency_matrix, constraint='degree'):
    """Computes the constraint matrix from a weighted adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : dense, sparse (N x N)
        weighted adjacency matrix.

    constraint : str, default = 'degree'
        ['identity'|'degree'|'similarity'|'dissimilarity']
        the type of constraint matrix to construct

    Returns
    -------
    D : array, sparse (N x N)
        constraint matrix
    """
    if constraint in ['degree']:
        D = spdiags(data=np.squeeze(np.asarray(adjacency_matrix.sum(axis=1))),
                    diags=[0], m=adjacency_matrix.shape[0],
                    n=adjacency_matrix.shape[0])

        return D
    elif constraint in ['identity']:
        D = eye(m=adjacency_matrix.shape[0], n=adjacency_matrix.shape[0], k=0)

        return D
    else:
        raise NotImplementedError('No other methods implemented.')


def create_laplacian(adjacency_matrix, laplacian='unnormalized'):
    """Computes the graph laplacian from a weighted adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : dense, sparse (N x N)
        weighted adjacency matrix.

    laplacian : str, default = 'unnormalized'
        ['normalized'|'unnormalized'|'randomwalk']
        the type of laplacian matrix to construct

    Returns
    -------
    L : sparse (N x N)
        Laplacian matrix.

    D : sparse (N x N)
        Diagonal degree matrix.

    References
    ----------
    sklearn - SpectralEmbedding
    megaman - spectral_embedding

    TODO: implement random walk laplacian
    TODO: implement renormalized laplacian
    """
    if laplacian in ['unnormalized']:
        laplacian_matrix, diagonal_matrix = \
            graph_laplacian(adjacency_matrix, normed=False, return_diag=True)

        diagonal_matrix = spdiags(data=diagonal_matrix, diags=[0],
                                  m=adjacency_matrix.shape[0],
                                  n=adjacency_matrix.shape[0])

        return laplacian_matrix, diagonal_matrix

    elif laplacian in ['normalized']:
        laplacian_matrix = graph_laplacian(adjacency_matrix,
                                           normed=True,
                                           return_diag=False)

        return laplacian_matrix

    else:
        raise ValueError('Unrecognized Graph Laplacian.')


def maximum(mata, matb):
    """This gives you the element-wise maximum between two sparse
    matrices of size (nxn)

    Reference
    ---------
        http://goo.gl/k0Yfmk
    """
    bisbigger = mata - matb
    bisbigger.data = np.where(bisbigger.data < 0, 1, 0)
    return mata - mata.multiply(bisbigger) + matb.multiply(bisbigger)

