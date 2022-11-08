"""Util.py has a modified class definition
we use to create pytorch readable files for
training/testing
"""
import os
from math import sqrt
import numpy as np
from scipy import stats
from torch_geometric.data import InMemoryDataset  # pylint: disable=import-error
from torch_geometric import data as DATA  # pylint: disable=import-error
import torch


class TestbedDataset(InMemoryDataset):
    """the class defined below is
    for testing the dataset
    """

    def __init__(self, root="/tmp", dataset="davis",
                 x_d=None, x_t=None, y_affinity=None,
                 transform=None, pre_transform=None, smile_graph=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print(
                "Pre-processed data found: {}, loading ...".format(
                    self.processed_paths[0]
                )
            )
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(
                "Pre-processed data {} not found, doing pre-processing...".format(
                    self.processed_paths[0]
                )
            )
            self.process(x_d, x_t, y_affinity, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """returns raw_file_names
        """
        # pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        """returns the processes file names
        """
        return [self.dataset + ".pt"]

    def download(self):
        """used to download
        """
        # Download to `self.raw_dir`.
        # pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # x_d - list of SMILES, x_t: list of encoded target (categorical or one-hot),
    # y_affinity: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, x_d, x_t, y_affinity, smile_graph):
        """the main processing function
        """
        assert len(x_d) == len(x_t) and len(x_t) == len(
            y_affinity
        ), "The three lists must be the same length!"
        data_list = []
        data_len = len(x_d)
        for i in range(data_len):
            print("Converting SMILES to graph: {}/{}".format(i + 1, data_len))
            smiles = x_d[i]
            target = x_t[i]
            labels = y_affinity[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            gcn_data = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose( # pylint: disable=no-member
                    1, 0
                ),  # pylint: disable=no-member
                y_affinity=torch.FloatTensor([labels]),  # pylint: disable=no-member
            )
            gcn_data.target = torch.LongTensor([target])  # pylint: disable=no-member
            gcn_data.__setitem__("c_size", torch.LongTensor([c_size])) # pylint: disable=no-member
            # append graph, label and target sequence to data list
            data_list.append(gcn_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print("Graph construction done. Saving to file.")
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


def rmse(y_affinity, var_f):
    """root mean squared deviation
    """
    return sqrt(((y_affinity - var_f) ** 2).mean(axis=0))


def mse(y_affinity, var_f):
    """mean squred error
    """
    return ((y_affinity - var_f) ** 2).mean(axis=0)


def pearson(y_affinity, var_f):
    """calculating pearson's coefficient
    """
    corr_coeff = np.corrcoef(y_affinity, var_f)[0, 1]
    return corr_coeff


def spearman(y_affinity, var_f):
    """calculating spearman's rho
    """
    spear_rho = stats.spearmanr(y_affinity, var_f)[0]
    return spear_rho


def c_i(y_affinity, var_f):
    """comment explaining this function
    """
    ind = np.argsort(y_affinity)
    y_affinity = y_affinity[ind]
    var_f = var_f[ind]
    i = len(y_affinity) - 1
    j = i - 1
    var_z = 0.0
    snake_casing = 0.0
    while i > 0:
        while j >= 0:
            if y_affinity[i] > y_affinity[j]:
                var_z = var_z + 1
                cond_u = var_f[i] - var_f[j]
                if cond_u > 0:
                    snake_casing = snake_casing + 1
                elif cond_u == 0:
                    snake_casing = snake_casing + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    return snake_casing / var_z
