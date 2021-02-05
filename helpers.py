import numpy as np
import scipy
import shutil
import os
import torch
import traceback
import sys

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# Helper functions
def is_function(x):
    import types
    return isinstance(x, types.FunctionType) \
        or isinstance(x, types.BuiltinFunctionType)

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.filepath = filepath

    def write(self, message):
        with open (self.filepath, "a", encoding = 'utf-8') as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def mean_std(x, n=3):
    x = np.array(x)
    return '%.3f ± %.3f' % (x.mean(), x.std())

def min_max_scaling_from_0_to_1_LOGMAG(X):
    """
    normalizing function to be used ONLY with
    x -= NORM_METHOD(x)
    :param X: array like
    :return:

    """
    X_offset = X - np.amin(X)
    X_offset /= np.amax(X_offset)
    return + X - (X_offset)

def get_list_of_files(dirName):
    '''
        For the given path, get the List of all files in the directory tree
    '''
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def get_count_of_audiofiles(dirName):
    list_wav = [k for k in get_list_of_files(dirName) if '.wav' in k]
    return np.array(list_wav).shape[0]

def mahalanobis_distance(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = scipy.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


def max_finder(lst, N, result_lst = list()):
    while N > 0:
        max_value = np.amax(lst)
        result_lst.append(max_value)
        lst = np.delete(lst, np.argmax(lst))
        N-=1
    return result_lst


def min_finder(lst, N, result_lst = list()):
    while N > 0:
        min_value = min(lst)
        result_lst.append(min_value)
        lst.remove(min_value)
        N-=1
    return result_lst


def create_folder(fd, add=0):
    import os
    if add != 0:
        if fd[-1].isdigit():
            le = len(fd.split('_')[-1])
            fd = fd[:-le] + str(add)
        else:
            fd = fd + "_" + str(add)
    if not os.path.exists(fd):
        os.makedirs(fd)
        return fd
    else:
        fd = create_folder(fd, add=add+1)
        return fd


def delete_folder(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)


def check_folder(dirpath):
    return os.path.exists(dirpath) and os.path.isdir(dirpath)



import seaborn as sns
import matplotlib.pyplot as plt
def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            vertical=True)
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.legend(legends)
