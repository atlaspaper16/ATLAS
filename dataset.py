import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data.data import DataEdgeAttr # Changed this import
from logger import Logger
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import GlobalStorage
add_safe_globals([GlobalStorage])
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr]) # Added DataTensorAttr here

from torch_geometric.datasets import (
    Amazon, Coauthor, HeterophilousGraphDataset, WikiCS, LINKXDataset, DeezerEurope,
    AttributedGraphDataset, Actor, WebKB, WikipediaNetwork, Twitch,
    Yelp, Flickr, AmazonProducts, Reddit2  # NEW
)


import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS, LINKXDataset, DeezerEurope, AttributedGraphDataset, Actor, WebKB, WikipediaNetwork, Twitch
#from ogb.nodeproppred import NodePropPredDataset

import numpy as np
import scipy.sparse as sp
from os import path
#from google_drive_downloader import GoogleDriveDownloader as gdd
import gdown
import scipy
from torch_geometric.datasets import Planetoid
from data_utils import * # dataset_drive_url, rand_train_test_idx, even_quantile_labels
from torch_geometric.data import Data
import os
import json
import pandas as pd
from scipy.sparse import coo_matrix
import csv




class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None
        self.num_nodes = 0

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def is_twitch_dataset(name):
    return name.upper() in {'DE','EN','ES','FR','PT','RU'}



def is_attributed_graph_dataset(name):
    attributed_names = {
        #'actor', 'cornell', 'texas', 'wisconsin'  # Explicit known names from AttributedGraphDataset
        "Wiki", "Cora" "CiteSeer", "PubMed", "BlogCatalog", "PPI", "Flickr", "Facebook", "Twitter", "TWeibo", "MAG"
    }
    return name in attributed_names

def is_linkx_dataset(name):
    linkx_names = {
        'arxiv-year', 'genius', 'fb100',
        'penn94', 'twitch-gamer', 'wiki'
    }
    return name in linkx_names

def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset
        Returns NCDataset
    """
    print(dataname)
    if dataname in  ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, dataname)
    elif dataname in  ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname)
    elif dataname in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = load_hetero_dataset(data_dir, dataname)
    elif dataname == 'wikics':
        dataset = load_wikics_dataset(data_dir)
    elif dataname in ('ogbn-arxiv', 'ogbn-products','ogbn-papers100M'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname)
    elif dataname in ('chameleon-filtered', 'squirrel-filtered'):
        dataset = load_geom_filtered_dataset(data_dir, dataname)
    elif is_linkx_dataset(dataname):
        dataset = load_linkx_dataset(data_dir, dataname)
    elif dataname == 'deezer-europe':
        dataset = load_deezer_europe(data_dir)
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat(data_dir)
    elif dataname == 'actor':
        dataset = load_actor_dataset(data_dir)
    elif is_attributed_graph_dataset(dataname):
        dataset = load_attributed_graph_dataset(data_dir, dataname)
    elif dataname in ('cornell', 'texas', 'wisconsin'):
        dataset = load_webkb_dataset(data_dir, dataname)
    elif dataname in ('chameleon', 'squirrel'):
        dataset = load_wikipedia_network_dataset(data_dir, dataname)
    elif dataname == 'BGP':
        dataset = load_bgp_dataset(data_dir)
    elif dataname.startswith('twitch-'):
        dataset = load_twitch_dataset(data_dir, dataname)
    elif dataname == 'wiki-cooc':
        return load_wiki_cooc_dataset(data_dir, dataname)
    elif dataname in ('yelp', 'flickr', 'amazon-products', 'reddit2'):
        dataset = load_graphsaint_dataset(data_dir, dataname)

    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_graphsaint_dataset(data_dir, dataname):
    """
    Loads GraphSAINT-style large node classification datasets from PyG:
    yelp, flickr, amazon-products, reddit2
    """
    mapping = {
        'yelp': Yelp,
        'flickr': Flickr,
        'amazon-products': AmazonProducts,
        'reddit2': Reddit2,
    }
    cls = mapping[dataname]
    torch_dataset = cls(root=f'{data_dir}/{dataname}')
    data = torch_dataset[0]

    # Ensure dense float features when needed (Flickr can be sparse in some envs)
    x = data.x.to_dense().float() if hasattr(data.x, 'to_dense') else data.x.float()

    dataset = NCDataset(dataname)
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    dataset.num_nodes = data.num_nodes
    return dataset


def load_bgp_dataset(data_dir):
    dataset = NCDataset('bgp')

    folder = path.join(data_dir, 'BGP')
    edge_index = torch.tensor(np.load(path.join(folder, 'edge_index.npy')), dtype=torch.long)
    if edge_index.dim() == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.T  # Ensure shape (2, E)

    node_feat = torch.tensor(np.load(path.join(folder, 'x.npy')), dtype=torch.float)
    label = torch.tensor(np.load(path.join(folder, 'y.npy')), dtype=torch.long)

    dataset.graph = {
        'edge_index': edge_index,
        'node_feat': node_feat,
        'edge_feat': None,
        'num_nodes': node_feat.size(0)
    }
    dataset.label = label
    dataset.num_nodes = node_feat.size(0)

    return dataset



def load_twitch(data_dir, lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"{data_dir}/twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    
    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0] # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label
    
    return A, label, features




def load_wiki_cooc_dataset(data_dir, name):
    file = os.path.join(data_dir, f'{name.replace("-", "_")}.npz')
    npz = np.load(file, allow_pickle=True)
    d = npz['arr_0'].item() if 'arr_0' in npz else dict(npz)

    x = torch.tensor(d['node_features'], dtype=torch.float)
    y = torch.tensor(d['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(d['edges'], dtype=torch.long).t().contiguous()

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': edge_index,
        'node_feat': x,
        'edge_feat': None,
        'num_nodes': x.size(0)
    }
    dataset.label = y
    dataset.num_nodes = x.size(0)

    return dataset



def load_twitch_dataset(data_dir, name):
    assert name.startswith("twitch-"), "Expected dataset name like 'twitch-en'"
    lang_code = name.split("-")[1].upper()
    assert lang_code in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), "Unsupported Twitch language code."

    folder = f"{data_dir}/twitch/{lang_code}"
    edge_file = f"{folder}/musae_{lang_code}_edges.csv"
    feature_file = f"{folder}/musae_{lang_code}_features.json"
    target_file = f"{folder}/musae_{lang_code}_target.csv"

    # Load edges
    A, label, features = load_twitch(data_dir, lang_code)
    dataset = NCDataset(lang_code)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)

    return dataset


def load_geom_filtered_dataset(data_dir, name):
    file = os.path.join(data_dir, f'{name}.npz')
    npz = np.load(file, allow_pickle=True)
    d = npz['arr_0'].item() if 'arr_0' in npz else dict(npz)

    x = torch.tensor(d['node_features'], dtype=torch.float)
    y = torch.tensor(d['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(d['edges'], dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': edge_index,
        'node_feat': x,
        'edge_feat': None,
        'num_nodes': x.size(0)
    }
    dataset.label = y
    dataset.num_nodes = x.size(0)
    return dataset


def load_wikipedia_network_dataset(data_dir, name):
    torch_dataset = WikipediaNetwork(root=f'{data_dir}/WikipediaNetwork', name=name, geom_gcn_preprocess=True)
    data = torch_dataset[0]

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    dataset.num_nodes = data.num_nodes

    return dataset




def load_webkb_dataset(data_dir, name):
    dataset_tg = WebKB(root=f'{data_dir}/webkb', name=name.capitalize())
    data = dataset_tg[0]
    dataset = NCDataset(name)

    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    dataset.num_nodes = data.num_nodes

    return dataset







def load_attributed_graph_dataset(data_dir, name):
    dataset = AttributedGraphDataset(name=name, root=f'{data_dir}/attributed-graphs')
    data = dataset[0]
    nc_dataset = NCDataset(name)

    if name.lower() == 'flickr' and hasattr(data.x, 'to_dense'):
        data.x = data.x.to_dense().float()

    nc_dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    nc_dataset.label = data.y
    nc_dataset.num_nodes = data.num_nodes

    return nc_dataset


def load_actor_dataset(data_dir):
    torch_dataset = Actor(root=f'{data_dir}/actor')
    data = torch_dataset[0]
    dataset = NCDataset('actor')

    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    dataset.num_nodes = data.num_nodes

    return dataset




def load_snap_patents_mat(data_dir, nclass=5):
    mat_path = f'{data_dir}/snap-patents/snap_patents.mat'
    if not path.exists(mat_path):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=p, output=mat_path, quiet=False)

    fulldata = scipy.io.loadmat(mat_path)

    dataset = NCDataset('snap-patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])

    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': node_feat,
        'num_nodes': num_nodes
    }

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)
    dataset.num_nodes = num_nodes

    return dataset

def load_linkx_dataset(data_dir, name):
    torch_dataset = LINKXDataset(name=name, root=f'{data_dir}/linkx/')
    data = torch_dataset[0]

    edge_index = data.edge_index

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    return dataset


def load_deezer_europe(data_dir):
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{data_dir}/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset






def load_planetoid_dataset(data_dir, name, no_feat_norm=True):
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    print(f"Num nodes: {data.num_nodes}")

    dataset = NCDataset(name)
    dataset.num_nodes = data.num_nodes

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': data.num_nodes}
    dataset.label = label

    return dataset


def load_wiki_new(data_dir, name):
    path= f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz'
    data=np.load(path)
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat=data['node_features'] # unnormalized
    labels=data['node_labels']
    edges=data['edges'] #(E, 2)
    edge_index=edges.T

    dataset = NCDataset(name)

    edge_index=torch.as_tensor(edge_index)
    node_feat=torch.as_tensor(node_feat)
    labels=torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels
    dataset.num_nodes = data.num_nodes

    return dataset

def load_wikics_dataset(data_dir):
    wikics_dataset = WikiCS(root=f'{data_dir}/wikics/')
    data = wikics_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset('wikics')
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_nodes = data.num_nodes

    return dataset

def load_hetero_dataset(data_dir, name):
    #transform = T.NormalizeFeatures()
    torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    ## dataset splits are implemented in data_utils.py
    
    #dataset.train_idx = torch.where(data.train_mask[:,0])[0]
    #dataset.valid_idx = torch.where(data.val_mask[:,0])[0]
    #dataset.test_idx = torch.where(data.test_mask[:,0])[0]
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]



    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_nodes = data.num_nodes

    return dataset


def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                                 name='Photo', transform=transform)
    elif name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                                 name='Computers', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_nodes = data.num_nodes

    return dataset

def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
    elif name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    dataset.num_nodes = data.num_nodes

    return dataset

def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    pyg_dataset = PygNodePropPredDataset(name=name, root=f'{data_dir}/ogb')

    pyg_data = pyg_dataset[0]  # This is a torch_geometric.data.Data object
    split_idx = pyg_dataset.get_idx_split()

    # Extract edge_index, node features, labels, num nodes
    edge_index = pyg_data.edge_index
    node_feat = pyg_data.x
    labels = pyg_data.y
    num_nodes = pyg_data.num_nodes

    # Store in our custom format
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,  # OGB usually doesn't provide edge features
        'node_feat': node_feat,
        'num_nodes': num_nodes,
    }

    dataset.label = labels.reshape(-1, 1)

    def ogb_idx_to_tensor():
        return {k: torch.as_tensor(v) for k, v in split_idx.items()}

    dataset.load_fixed_splits = ogb_idx_to_tensor
    dataset.num_nodes = num_nodes

    return dataset



def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/pokec/")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    dataset.num_nodes = num_nodes
    return dataset
