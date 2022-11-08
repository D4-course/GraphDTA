"""python file responsible for creating data
"""
import pandas as pd
import numpy as np
# import os
# import json, pickle
# from collections import OrderedDict
from rdkit import Chem # pylint: disable=import-error
from rdkit.Chem import MolFromSmiles # pylint: disable=import-error
import networkx as nx # pylint: disable=import-error
from utils import *

SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOC)}
SEQ_DICT_LEN = len(SEQ_DICT)
MAX_SEQ_LEN = 1000


def atom_features(atom):
    """featurizing the atom? mugundan help
    """
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + [atom.GetIsAromatic()]
    )


def one_of_k_encoding(var_x, allowable_set):
    """one of k enconding
    """
    if var_x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(var_x, allowable_set))
    return list(map(lambda s: var_x == s, allowable_set))


def one_of_k_encoding_unk(var_x, allowable_set):
    """Maps inputs not in the
    allowable set to the last element.
    """
    if var_x not in allowable_set:
        var_x = allowable_set[-1]
    return list(map(lambda s: var_x == s, allowable_set))


def smile_to_graph(smile):
    """converts smiles to graph
    """
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    smile_graph = nx.Graph(edges).to_directed()
    edge_index = []
    for e_one, e_two in smile_graph.edges:
        edge_index.append([e_one, e_two])

    return c_size, features, edge_index


def seq_cat(prot):
    """seq cat function
    """
    np_x = np.zeros(MAX_SEQ_LEN)
    for i, c_h in enumerate(prot[:MAX_SEQ_LEN]):
        np_x[i] = SEQ_DICT[c_h]
    return np_x


compound_iso_smiles = []
for dt_name in ["kiba", "davis"]:
    opts = ["train", "test"]
    for opt in opts:
        df = pd.read_csv("./data/" + dt_name + "_" + opt + ".csv")
        compound_iso_smiles += list(df["compound_iso_smiles"])
compound_iso_smiles = set(compound_iso_smiles)
SMILE_GRAPH = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    SMILE_GRAPH[smile] = g


def verify_smile(smile):
    """used for verifying mol from smiles
    """
    return not MolFromSmiles(smile) is None


def create_test(iso_smile, target):
    """create test data
    taking smiles and target protein
    """
    # convert to PyTorch data format
    # unused variable incoming
    processed_data_file_test = "./data/processed/test.pt" # pylint: disable=unused-variable
    test_drugs = [iso_smile]
    test_prots = [target]
    test_y = [1]
    x_t = [seq_cat(t) for t in test_prots]
    test_drugs, test_prots, test_y = (
        np.asarray(test_drugs),
        np.asarray(x_t),
        np.asarray(test_y),
    )
    # make data PyTorch Geometric ready
    print("preparing test.pt in pytorch format!")
    # unused variable
    test_data = TestbedDataset( # pylint: disable=unused-variable
        root="data",
        dataset="test",
        x_d=test_drugs,
        x_t=test_prots,
        y_affinity=test_y,
        smile_graph=SMILE_GRAPH,
    )
