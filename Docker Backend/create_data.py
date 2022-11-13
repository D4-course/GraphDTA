"""
File responsible for converting processing the davis and kiba
datasets as well as for converting the user input into a 
pytorch file for prediction purposes
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
    """
    Converts the features of the atoms into a form 
    that can be made into a graph 
    implementation to run graph algos on
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
    """
    Subfunction of atom features
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
    """
    Crops protein sequence and does mapping
    """
    np_x = np.zeros(MAX_SEQ_LEN)
    for i, c_h in enumerate(prot[:MAX_SEQ_LEN]):
        np_x[i] = SEQ_DICT[c_h]
    return np_x

"""
Reads the kiba and davis .csv files 
to make a smiles graph used in creating the pytorch files

This is automatically run when the folder is imported
"""
COMPOUND_ISO_SMILES = []
for dt_name in ["kiba", "davis"]:
    opts = ["train", "test"]
    for opt in opts:
        df = pd.read_csv("./data/" + dt_name + "_" + opt + ".csv")
        COMPOUND_ISO_SMILES += list(df["compound_iso_smiles"])
COMPOUND_ISO_SMILES = set(COMPOUND_ISO_SMILES)
SMILE_GRAPH = {}
for smile_seq in COMPOUND_ISO_SMILES:
    g = smile_to_graph(smile_seq)
    SMILE_GRAPH[smile_seq] = g


def verify_smile(smile):
    """
    Verify if the input sequence is a smile
    """
    return not MolFromSmiles(smile) is None

def verify_protein(prot):
    """
    Verify if the sequence is a protein
    """
    formatted_prot = prot.translate(str.maketrans("", "", " \n\t")).upper()
    return set(formatted_prot).issubset(set("ACDEFGHIKLMNPQRSTVWXY"))

def create_test(iso_smile, target):
    """
    Taking user input, create a pytorch file of the 
    data to be tested
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
