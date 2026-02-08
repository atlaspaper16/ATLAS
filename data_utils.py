import torch
import torch.nn.functional as F
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS, LINKXDataset, DeezerEurope, Twitch, WikipediaNetwork
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import pickle
import random
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (
    HeterophilousGraphDataset, WikiCS, LINKXDataset, DeezerEurope, Twitch, WikipediaNetwork, Actor,
    Yelp, Flickr, AmazonProducts, Reddit2  # NEW
)



import torch
import cudf, cugraph
import random

# ==============================
# Connected Components Count (cuGraph API)
# ==============================

def connected_components_count(G):
    try:
        df = cugraph.components.connected_components(G)
    except AttributeError:
        df = cugraph.connected_components(G)

    label_col = 'labels' if 'labels' in df.columns else ('component' if 'component' in df.columns else None)
    if label_col is None:
        raise RuntimeError(f"Unexpected CC output columns: {df.columns}")
    return int(df[label_col].nunique()), df, label_col

# ==============================
# Build Graph from PyTorch edge_index
# ==============================

def build_cugraph_from_edge_index(edge_index: torch.Tensor, directed: bool = False):
    src, dst = edge_index
    gdf = cudf.DataFrame({
        'src': src.cpu().numpy(),
        'dst': dst.cpu().numpy()
    })
    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(gdf, source='src', destination='dst')
    return G





def cap_val_test_overflow_to_train(train_mask, val_mask, test_mask,
                                   train_prop=0.6, valid_prop=0.2,
                                   seed=42):
    """
    Enforce target counts by moving overflow from val/test into train.
    Keeps behavior deterministic with a fixed seed.
    """
    assert train_mask.dtype == torch.bool
    assert val_mask.dtype == torch.bool
    assert test_mask.dtype == torch.bool
    assert train_mask.numel() == val_mask.numel() == test_mask.numel()

    num_nodes = train_mask.numel()
    n_train = int(train_prop * num_nodes)
    n_val   = int(valid_prop * num_nodes)
    n_test  = num_nodes - (n_train + n_val)

    g = torch.Generator(device=train_mask.device)
    g.manual_seed(seed)

    # ---- cap validation ----
    val_idx = val_mask.nonzero(as_tuple=False).view(-1)
    if val_idx.numel() > n_val:
        perm = torch.randperm(val_idx.numel(), generator=g, device=val_idx.device)
        overflow = val_idx[perm[n_val:]]   # extra nodes beyond target
        val_mask[overflow] = False
        train_mask[overflow] = True

    # ---- cap test ----
    test_idx = test_mask.nonzero(as_tuple=False).view(-1)
    if test_idx.numel() > n_test:
        perm = torch.randperm(test_idx.numel(), generator=g, device=test_idx.device)
        overflow = test_idx[perm[n_test:]] # extra nodes beyond target
        test_mask[overflow] = False
        train_mask[overflow] = True

    # Recompute indices for printing / downstream use
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    val_idx   = val_mask.nonzero(as_tuple=False).view(-1)
    test_idx  = test_mask.nonzero(as_tuple=False).view(-1)

    return train_mask, val_mask, test_mask, train_idx, val_idx, test_idx, (n_train, n_val, n_test)



def rand_train_test_idx(label, train_prop=0.6, valid_prop=0.2, ignore_negative=True, pkl_path=None):
    """
    Splits nodes into train/val/test sets. Supports optional community-based splitting.

    Args:
        label (Tensor): Node labels.
        train_prop (float): Proportion of training nodes.
        valid_prop (float): Proportion of validation nodes.
        ignore_negative (bool): Skip nodes with label -1.
        pkl_path (str or None): If provided, loads communities for intra-community splits.

    Returns:
        train_idx, val_idx, test_idx (Tensor): Index tensors.
    """
    num_nodes = label.size(0)

    if pkl_path is not None:
        with open(pkl_path, 'rb') as f:
            community_partitions = pickle.load(f)
        print(f"Community paritions: {len(community_partitions)}")
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for i, community in enumerate (community_partitions):
            if not community:
                continue

            random.shuffle(community)
            n = len(community)
            n_train = int(train_prop * n)
            n_val = int(valid_prop * n)
            if(i%2 ==0):
                n_val = n_val + 1
              

            train_mask[community[:n_train]] = True
            val_mask[community[n_train:n_train + n_val]] = True
            test_mask[community[n_train + n_val:]] = True


        train_idx = train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = val_mask.nonzero(as_tuple=False).view(-1)
        test_idx = test_mask.nonzero(as_tuple=False).view(-1)

        train_mask, val_mask, test_mask, train_idx, val_idx, test_idx, targets = cap_val_test_overflow_to_train(
                train_mask, val_mask, test_mask,
                train_prop=0.6, valid_prop=0.2, seed=42
            )   


    else:
        labeled_nodes = torch.where(label != -1)[0] if ignore_negative else torch.arange(num_nodes)
        n = labeled_nodes.size(0)

        n_train = int(train_prop * n)
        n_val = int(valid_prop * n)

        perm = torch.as_tensor(np.random.permutation(n))
        train_idx = labeled_nodes[perm[:n_train]]
        val_idx = labeled_nodes[perm[n_train:n_train + n_val]]
        test_idx = labeled_nodes[perm[n_train + n_val:]]


        # Print the sizes
    print(f"Size of the training set: {len(train_idx)}")
    print(f"Size of the validation set: {len(val_idx)}")
    print(f"Size of the test set: {len(test_idx)}")


    return train_idx, val_idx, test_idx



def is_linkx_dataset():
    linkx_names = {
        'arxiv-year', 'genius', 'fb100',
        'penn94', 'twitch-gamer', 'wiki'
    }
    return linkx_names



def load_fixed_splits(data_dir, dataset, name, pkl_path=None):
    splits_lst = []
    print("Load Fixed splits")
    if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            splits_lst.append(splits)
    elif name in ['wikics']:
        torch_dataset = WikiCS(root=f"{data_dir}/wikics/")
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(torch.logical_or(data.val_mask, data.stopping_mask)[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:])[0]
            splits_lst.append(splits)
    elif name in ['amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics']:
        splits = {}
        idx = np.load(f'{data_dir}/{name}_split.npz')
        splits['train'] = torch.from_numpy(idx['train'])
        splits['valid'] = torch.from_numpy(idx['valid'])
        splits['test'] = torch.from_numpy(idx['test'])
        splits_lst.append(splits)
    elif name in ['pokec']:
        split = np.load(f'{data_dir}/{name}/{name}-splits.npy', allow_pickle=True)
        for i in range(split.shape[0]):
            splits = {}
            splits['train'] = torch.from_numpy(np.asarray(split[i]['train']))
            splits['valid'] = torch.from_numpy(np.asarray(split[i]['valid']))
            splits['test'] = torch.from_numpy(np.asarray(split[i]['test']))
            splits_lst.append(splits)
    elif name in ["chameleon", "squirrel"]:
        file_path = f"{data_dir}/geom-gcn/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N), 10 splits
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)


    elif name in is_linkx_dataset():
        torch_dataset = LINKXDataset(name=name, root=f'{data_dir}/linkx/')
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {
                'train': torch.where(data.train_mask[:, i])[0],
                'valid': torch.where(data.val_mask[:, i])[0],
                'test': torch.where(data.test_mask[:, i])[0],
            }
            splits_lst.append(splits)
    elif name in ['deezer-europe']:
        deezer = scipy.io.loadmat(f'{data_dir}/deezer-europe.mat')
        A, label, features = deezer['A'], deezer['label'], deezer['features']
        num_nodes = label.shape[0]
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        node_feat = torch.tensor(features.todense(), dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long).squeeze()

        dataset.graph = {
            'edge_index': edge_index,
            'edge_feat': None,
            'node_feat': node_feat,
            'num_nodes': num_nodes
        }
        dataset.label = label
        split = rand_train_test_idx(label, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
        splits_lst.append({"train": split[0], "valid": split[1], "test": split[2]})
    elif name in ['snap-patents']:
        splits_lst = []
        for _ in range(1):
            split = {}
            train_idx, val_idx, test_idx = rand_train_test_idx(dataset.label, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
            split['train'] = train_idx
            split['valid'] = val_idx
            split['test'] = test_idx
            splits_lst.append(split)
        return splits_lst
    elif name in ["Wiki", "BlogCatalog", "PPI", "Flickr", "Facebook", "Twitter", "TWeibo"]:
        for _ in range(1):
            split = {}
            train_idx, val_idx, test_idx = rand_train_test_idx(dataset.label, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
            split['train'] = train_idx
            split['valid'] = val_idx
            split['test'] = test_idx
            splits_lst.append(split)
    elif name in ['actor']:
        data = Actor(root=f"{data_dir}/actor")[0]
        label = data.y
        for _ in range(1):
            train_idx, valid_idx, test_idx = rand_train_test_idx(label, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
            splits = {
                'train': train_idx,
                'valid': valid_idx,
                'test': test_idx
            }   
            splits_lst.append(splits)
    
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset_tg = WebKB(root=f"{data_dir}/webkb", name=name.capitalize())
        data = dataset_tg[0]
        label = data.y
        for _ in range(1):
            train_idx, valid_idx, test_idx = rand_train_test_idx(label, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
            splits = {
                'train': train_idx,
                'valid': valid_idx,
                'test': test_idx
            }
            splits_lst.append(splits)

    elif name in ['chameleon-filtered', 'squirrel-filtered']:
        file_path = f"{data_dir}/{name}.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # shape: (10, N)
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)
            print(
                f"Split {i}: "
                f"Train={len(splits['train'])}, "
                f"Valid={len(splits['valid'])}, "
                f"Test={len(splits['test'])}"
            )

    elif name in ['chameleon', 'squirrel']:
        torch_dataset = WikipediaNetwork(root=f'{data_dir}/WikipediaNetwork', name=name, geom_gcn_preprocess=True)
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits["train"] = torch.where(data.train_mask[:, i])[0]
            splits["valid"] = torch.where(data.val_mask[:, i])[0]
            splits["test"] = torch.where(data.test_mask[:, i])[0]
            splits_lst.append(splits)
    elif name == 'BGP':
        splits_lst = []
        for _ in range(1):  # Create 10 random splits
            split = {}
            train_idx, val_idx, test_idx = rand_train_test_idx(dataset.label, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
            split['train'] = train_idx
            split['valid'] = val_idx
            split['test'] = test_idx
            splits_lst.append(split)
    
    elif name == 'wiki-cooc':
        splits_lst = []
        for _ in range(1):  # Create 10 random splits
            split = {}
            train_idx, val_idx, test_idx = rand_train_test_idx(dataset.label, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
            split['train'] = train_idx
            split['valid'] = val_idx
            split['test'] = test_idx
            splits_lst.append(split)


    elif name.startswith('twitch-'):
        lang_code = name.split("-")[1].upper()
        assert lang_code in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), "Unsupported Twitch language code."

        target_path = f"{data_dir}/twitch/{lang_code}/musae_{lang_code}_target.csv"
        label = np.genfromtxt(target_path, delimiter=',', skip_header=1, usecols=1)  # Skip node ID column
        label_tensor = torch.tensor(label, dtype=torch.long)

        for _ in range(1):  # Generate 10 random splits
            train_idx, val_idx, test_idx = rand_train_test_idx(label_tensor, train_prop=0.6, valid_prop=0.2, pkl_path=pkl_path)
            splits_lst.append({'train': train_idx, 'valid': val_idx, 'test': test_idx})

        # NEW: OGB Node Property Prediction datasets
    elif name.startswith("ogbn-"):
        pyg_dataset = PygNodePropPredDataset(name=name, root=f"{data_dir}/ogb")
        
        split_idx = pyg_dataset.get_idx_split()
        splits = {
            "train": torch.as_tensor(split_idx["train"]),
            "valid": torch.as_tensor(split_idx["valid"]),
            "test": torch.as_tensor(split_idx["test"]),
        }
        splits_lst.append(splits)
        
    elif name in ['yelp', 'flickr', 'amazon-products', 'reddit2']:
        # Use the official GraphSAINT/SGC splits provided by PyG
        mapping = {
            'yelp': Yelp,
            'flickr': Flickr,
            'amazon-products': AmazonProducts,
            'reddit2': Reddit2,
        }
        torch_dataset_cls = mapping[name]
        torch_dataset = torch_dataset_cls(root=f"{data_dir}/{name}")
        data = torch_dataset[0]

        # Each has boolean train/val/test masks; convert to index lists
        splits = {
            'train': torch.where(data.train_mask)[0],
            'valid': torch.where(data.val_mask)[0],
            'test': torch.where(data.test_mask)[0],
        }
        splits_lst.append(splits)


    else:
        raise NotImplementedError


    return splits_lst

def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list)/len(acc_list)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

# in data_utils.py
def eval_rocauc(y_true, y_pred):
    """
    Binary ROC-AUC (for Minesweeper & Questions):
      - y_true can be [N], [N,1], or one-hot [N,2]
      - y_pred are logits: [N,1] or [N,2]
      - computes AUC for the POSITIVE class (class=1)
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    # Normalize labels -> [N] in {0,1}
    if y_true.dim() == 2 and y_true.size(1) == 2:   # one-hot
        y_true = y_true.argmax(dim=1)
    elif y_true.dim() == 2 and y_true.size(1) == 1:
        y_true = y_true.squeeze(1)

    # Positive-class probability from logits
    if y_pred.dim() == 2 and y_pred.size(1) == 2:
        p = F.softmax(y_pred, dim=-1)[:, 1].numpy()
    else:
        p = torch.sigmoid(y_pred.view(-1)).numpy()

    return roc_auc_score(y_true.numpy(), p)




dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}
