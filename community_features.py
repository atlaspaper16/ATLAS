import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain
import pickle
import cudf
import cugraph
import cupy
import cugraph
import os
import torch.nn as nn
import gc
import statistics
import os
from collections import defaultdict
import math
import random

# ========== GPU UTILS ==========
import collections
has_gpu = False
try:

    has_gpu = True
except ImportError:
    pass






def build_cugraph(data):
    edge_index = data.graph['edge_index']
    src, dst = edge_index
    gdf = cudf.DataFrame({
        'src': src.cpu().numpy(),
        'dst': dst.cpu().numpy()
    })
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(gdf, source='src', destination='dst')
    return G



def compute_nmi_from_partition_and_labels(partition, labels_np, return_stats: bool = False):
    """
    Compute NMI(C, L) between community partition (C) and labels (L) as in ATLAS:
        NMI = 2 * I(L; C) / (H(L) + H(C)).

    If return_stats=True, returns a dict with:
        {"nmi": ..., "I": ..., "H_L": ..., "H_C": ...}
    else returns only the NMI float (backward compatible).

    partition: list/1D array of community IDs (len = num_nodes)
    labels_np: 1D numpy array of integer class labels (len = num_nodes)
    """
    partition = np.asarray(partition)
    labels_np = np.asarray(labels_np)
    assert partition.shape[0] == labels_np.shape[0]

    n = partition.shape[0]

    # Map labels and communities to compact indices
    label_vals, label_idx = np.unique(labels_np, return_inverse=True)
    comm_vals, comm_idx = np.unique(partition, return_inverse=True)
    m = label_vals.shape[0]
    k = comm_vals.shape[0]

    # Contingency table n_ij
    n_ij = np.zeros((m, k), dtype=np.float64)
    for i in range(n):
        n_ij[label_idx[i], comm_idx[i]] += 1.0

    # Marginals
    n_i = n_ij.sum(axis=1)  # label counts
    n_j = n_ij.sum(axis=0)  # community counts

    # Probabilities
    p_ij = n_ij / n
    p_i = n_i / n
    p_j = n_j / n

    # Mutual information I(L;C)
    I = 0.0
    for a in range(m):
        for b in range(k):
            if p_ij[a, b] > 0:
                I += p_ij[a, b] * math.log(p_ij[a, b] / (p_i[a] * p_j[b]))

    # Entropies H(L), H(C)
    H_L = -sum(pi * math.log(pi) for pi in p_i if pi > 0)
    H_C = -sum(pj * math.log(pj) for pj in p_j if pj > 0)

    denom = H_L + H_C
    nmi_val = 0.0 if denom <= 0 else 2.0 * I / denom

    print(
        f"    [NMI] I(L;C)={I:.6f}, "
        f"H(L)={H_L:.6f}, H(C)={H_C:.4f}, "
        f"H(L)+H(C)={H_L + H_C:.4f}, "
        f"NMI={nmi_val:.4f}"
    )

    if return_stats:
        return {"nmi": nmi_val, "I": I, "H_L": H_L, "H_C": H_C}
    return nmi_val


def theorem1_refinement_check(base_stats: dict, new_stats: dict):
    """
    Theorem-1 condition check (Section 2, Theorem 1 in ATLAS):

        ΔI / ΔH  >  NMI(C;L)/2

    where:
        ΔI = I(C';L) - I(C;L)
        ΔH = H(C')   - H(C)

    Notes:
    - We use H(C) = H_C returned above (community entropy).
    - The theorem assumes ΔH > 0 (non-degenerate refinement). If ΔH <= 0, we mark FAIL.
    """
    delta_I = new_stats["I"] - base_stats["I"]
    delta_H = new_stats["H_C"] - base_stats["H_C"]
    threshold = base_stats["nmi"] / 2.0

    if delta_H <= 0:
        ratio = float("inf") if delta_H == 0 else (delta_I / delta_H)
        return False, delta_I, delta_H, ratio, threshold

    ratio = delta_I / delta_H
    ok = ratio > threshold
    return ok, delta_I, delta_H, ratio, threshold

def run_louvain_gpu(G, num_nodes, resolution):
    #print(f"Running on GPU {num_nodes}")
    # Run Louvain community detection
    parts, mod = cugraph.louvain(G, resolution=resolution)

    # Sort by vertex ID to ensure alignment
    parts = parts.sort_values(by='vertex').reset_index(drop=True)

    # Initialize all nodes to -1 (unassigned)
    partition_array = np.full(num_nodes, 0, dtype=np.int32)

    # Extract vertex-community mappings
    vertex_ids = parts['vertex'].to_pandas().values
    comm_ids = parts['partition'].to_pandas().values

    # Assign communities to the correct node indices
    partition_array[vertex_ids] = comm_ids

    partition = partition_array.tolist()

    return partition, mod

def cleanup_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

    cupy.get_default_memory_pool().free_all_blocks()

# ========== CPU UTILS ==========

def run_louvain_cpu(data, num_nodes, resolution):
    print("Running on CPU")
    edge_index = data.graph['edge_index'].cpu()
    src, dst = edge_index
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    partition_dict = community_louvain.best_partition(G, resolution=resolution)
    partition = [partition_dict[i] for i in range(num_nodes)]
    Q = community_louvain.modularity(partition_dict, G)
    return partition, Q

# ========== MODULARITY-GUIDED SEARCH ==========

def estimate_slope(tried):
    sorted_res = sorted(tried)
    if len(sorted_res) < 2:
        return -0.1
    elif len(sorted_res) == 2:
        r1, r2 = sorted_res[-2:]
        return (tried[r2] - tried[r1]) / (r2 - r1)
    else:
        r1, r2, r3 = sorted_res[-3:]
        q1, q2, q3 = tried[r1], tried[r2], tried[r3]
        x = np.array([r1, r2, r3])
        y = np.array([q1, q2, q3])
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]




def find_adaptive_resolutions(
    data,
    device,
    min_modularity: float = 0.2,
    max_modularity_gap: float = 0.1,
    base_resolution: float = 0,
    enforce_theorem1: bool = True,
):
    """
    Modularity-guided resolution search, augmented with Theorem-1 refinement check.

    We fix the *base* partition C at `base_resolution` (default 0.1), then for every
    candidate resolution r (forming C'), we check whether:

        (ΔI / ΔH) > NMI(C;L)/2,

    where:
        ΔI = I(C';L) - I(C;L)
        ΔH = H(C')   - H(C)

    If `enforce_theorem1=True`, we still run the full search, but we APPLY the Theorem-1 test only at the end to filter the final `filtered_resolutions` list (post-search). If False, we only print PASS/FAIL and do not filter by Theorem-1.

    Returns:
        best_r, filtered_resolutions, partition_list, num_communities_list
    """
    delta_range = (0.1, 0.2)
    num_nodes = data.graph['num_nodes']
    tried, partitions = {}, {}
    theorem1_ok = {}
    theorem1_metrics = {}

    initial_res = [0.5, 1.0]
    print("[🔧] Building graph")

    # --- prepare labels for NMI (1D numpy array) ---
    labels = data.label
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
    labels_np = labels.cpu().numpy()

    # --- Build GPU graph once (if needed) ---
    G = None
    if device == 'cuda':
        G = build_cugraph(data)

    # --- Fixed base partition C at resolution=base_resolution ---
    if device == 'cuda':
        base_partition, base_Q = run_louvain_gpu(G, num_nodes, base_resolution)
    else:
        base_partition, base_Q = run_louvain_cpu(data, num_nodes, base_resolution)

    base_stats = compute_nmi_from_partition_and_labels(base_partition, labels_np, return_stats=True)
    print(
        f"[BASE C] res={base_resolution} → {max(base_partition)+1} communities "
        f"(Q={base_Q:.4f}, NMI={base_stats['nmi']:.4f})"
    )

    def eval_candidate(resolution: float):
        """Run Louvain at `resolution`, compute NMI stats, check Theorem-1, optionally keep."""
        if resolution in tried:
            return

        if device == 'cuda':
            partition, Q = run_louvain_gpu(G, num_nodes, resolution)
        else:
            partition, Q = run_louvain_cpu(data, num_nodes, resolution)

        new_stats = compute_nmi_from_partition_and_labels(partition, labels_np, return_stats=True)

        ok, dI, dH, ratio, thr = theorem1_refinement_check(base_stats, new_stats)
        theorem1_ok[resolution] = ok
        theorem1_metrics[resolution] = {'dI': dI, 'dH': dH, 'ratio': ratio, 'thr': thr}
        print(
            f"    [Thm1 vs base@{base_resolution}] res={resolution} | "
            f"ΔI={dI:.6f}, ΔH={dH:.6f}, ΔI/ΔH={ratio:.6f}, thr={thr:.6f}  =>  {'PASS' if ok else 'FAIL'}"
        )

        print(
            f"[✓] Resolution={resolution} → {max(partition)+1} communities "
            f"(Q={Q:.4f}, NMI={new_stats['nmi']:.4f})"
        )

        tried[resolution] = Q
        partitions[resolution] = partition

    # --- Initial resolutions ---
    for r in initial_res:
        eval_candidate(float(r))

    # --- Adaptive search loop ---
    while True:
        if not tried:
            break

        res_list = sorted(tried.keys())
        mods = [tried[r] for r in res_list]

        # stop if last modularity is too low
        if mods[-1] <= min_modularity and res_list[-1] > 1:
            print(f"[⛔] Modularity {mods[-1]:.4f} ≤ threshold {min_modularity}")
            break

        new_res = None

        # 1) Interpolation: check gaps between known resolutions
        for (r1, r2) in zip(res_list[:-1], res_list[1:]):
            Q1, Q2 = tried[r1], tried[r2]
            delta_q = abs(Q2 - Q1)
            if delta_q > max_modularity_gap:
                new_res = round((r1 + r2) / 2, 3)
                print(f"[➕] Interpolating {r1:.3f}–{r2:.3f} → {new_res} (ΔQ={delta_q:.4f})")
                break

        # 2) Extrapolation: extend beyond max resolution if modularity still good
        if new_res is None:
            max_res, max_Q = res_list[-1], mods[-1]
            if max_Q > min_modularity:
                delta_mod = np.random.uniform(*delta_range)
                Q_target = max_Q - delta_mod
                slope = estimate_slope(tried) or -0.05
                # if slope is near zero, avoid exploding step
                if abs(slope) < 1e-8:
                    slope = -0.05
                new_res = round(max_res + (Q_target - max_Q) / slope, 3)
                print(f"[🔍] Extrapolating beyond {max_res} → {new_res} targeting Q≈{Q_target:.4f}")

        # 3) Stop if no new resolution found
        if new_res is None or new_res in tried:
            break

        # Run candidate (this will apply Theorem-1 filtering if enabled)
        eval_candidate(float(new_res))

    # --- Cleanup GPU ---
    if device == 'cuda' and G is not None:
        try:
            del G
        except Exception:
            pass
        cleanup_gpu_memory()

    # --- Filter valid resolutions ---
    filtered_resolutions, partition_list, num_communities_list = [], [], []
    filtered_modularity = []
    modularity_dict = {}

    for r in sorted(tried.keys()):
        Q = tried[r]
        if Q >= min_modularity:
            # Apply Theorem-1 filtering ONLY at the end (post-search), if enabled
            if enforce_theorem1 and (not theorem1_ok.get(r, False)):
                m = theorem1_metrics.get(r, {})
                print(
                    f"[⚠️] Discarding {r:.3f} (fails Thm1 vs base@{base_resolution}: "
                    f"ΔI={m.get('dI', float('nan')):.6f}, ΔH={m.get('dH', float('nan')):.6f}, "
                    f"ΔI/ΔH={m.get('ratio', float('nan')):.6f} ≤ thr={m.get('thr', float('nan')):.6f})"
                )
                continue
            filtered_resolutions.append(r)
            filtered_modularity.append(Q)
            partition_list.append(partitions[r])
            num_communities_list.append(max(partitions[r]) + 1)
            modularity_dict[r] = Q
        else:
            print(f"[⚠️] Discarding {r:.3f} (Q={Q:.4f} < {min_modularity})")

    if not filtered_resolutions:
        print("[❌] No valid resolution found")
        return None, None, None, None

    # --- Choose best resolution (keeps your original selection rule) ---
    selected_initial = [r for r in filtered_resolutions if r >= 1 and r < 10]
    if len(selected_initial) < 1:
        best_r = min(filtered_resolutions)
    else:
        best_r = min(selected_initial)

    best_partition = partitions[best_r]

    # --- Save best partition ---
    communities = {}
    for node, cid in enumerate(best_partition):
        communities.setdefault(cid, []).append(node)
    comm_list = list(communities.values())

    with open(f"comm_res{best_r}.pkl", 'wb') as f:
        pickle.dump(comm_list, f)

    print(f"[📦] Saved best resolution {best_r} → {len(comm_list)} communities (Q={modularity_dict[best_r]:.4f})")
    print(f"[✅] Final Resolutions: {filtered_resolutions}")
    print(f"[✅] Final Modularity: {filtered_modularity}")

    if len(filtered_modularity) >= 2:
        diffs = [abs(filtered_modularity[i] - filtered_modularity[i+1]) for i in range(len(filtered_modularity)-1)]
        avg_gap = sum(diffs) / len(diffs)
        print(f"[✅] Used modularity {len(filtered_modularity)} Average modularity gap:{avg_gap}")
    else:
        print(f"[✅] Used modularity {len(filtered_modularity)} (avg gap undefined with <2 points)")

    return best_r, filtered_resolutions, partition_list, num_communities_list

def generate_louvain_embeddings(data, args, device='cuda'):
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    use_gpu = device == 'cuda'
    
    num_nodes = data.graph['num_nodes']
    #resolutions = None
    # === Parse resolutions ===
    print(args.res.lower() == 'none')
    if args.res.lower() == 'none':
        resolutions = None
    else:
        # e.g., input: --res 1.0 2.0 3.5
        resolutions = [float(r) for r in args.res.split()]
    
    embedding_dim = args.emb_dim
    del_q = args.del_q
    min_q = args.min_q

        # ---------------------------
    # 🔹 Check cache folder
    # ---------------------------
    """
    cache_dir = args.dataset
    cache_file = None
    
    if cache_dir is not None:
        os.makedirs("cache_dir", exist_ok=True)
        cache_file = os.path.join("cache_dir", cache_dir +"_louvain_cache.pt")

        if os.path.exists(cache_file):
            print(f"[💾] Loading cached Louvain embeddings from {cache_file}")
            cache = torch.load(cache_file, map_location=device)

            # Restore embeddings properly as ModuleList
            community_embeddings = nn.ModuleList([
                nn.Embedding(num_comms, embedding_dim, device=device)
                for num_comms in cache["num_communities_list"]
            ])
            for emb_layer, state in zip(community_embeddings, cache["community_embeddings"]):
                emb_layer.load_state_dict(state)

            return (cache["best_r"],
                    community_embeddings,
                    cache["all_community_ids"],
                    cache["num_communities_list"],
                    cache["resolutions"])

    """
    

    community_ids_per_res = []
    num_communities_list = []

    if resolutions is None:
        print("[⚙️] No resolutions provided. Using adaptive modularity-based selection...")
        best_r, resolutions, partition_list, _ = find_adaptive_resolutions(data, device,min_q,del_q)
        
        if resolutions is None:
            print("[❌] No valid resolution found")
            return None, None, None, [], -1


        for partition in partition_list:
            community_ids = torch.tensor([partition[n] for n in range(num_nodes)],
                                     dtype=torch.long, device=device)
            community_ids_per_res.append(community_ids)

    # Detach all after loop (only once)
        community_ids_per_res = [
            arr.detach() if arr.device == device else arr.clone().detach().to(device)
            for arr in community_ids_per_res
        ]

        num_communities_list = [arr.max().item() + 1 for arr in community_ids_per_res]


    elif len(resolutions) == 0:
        best_r = None
    else:
        best_r = None
        res_to_mod = {}
        res_to_partition = {}
        community_ids_per_res = []
        if use_gpu:
            G = build_cugraph(data)
        for res in resolutions:
            if use_gpu:
                partition, Q = run_louvain_gpu(G, num_nodes, res)
            else:
                partition, Q = run_louvain_cpu(data, num_nodes, res)

            res_to_mod[res] = Q
            res_to_partition[res] = partition

            community_ids = torch.tensor([partition[n] for n in range(num_nodes)],
                                 dtype=torch.long, device=device)
            community_ids_per_res.append(community_ids)

        community_ids_per_res = [
            arr.detach() if arr.device == device else arr.clone().detach().to(device)
            for arr in community_ids_per_res
        ]

        num_communities_list = [ids.max().item() + 1 for ids in community_ids_per_res]

        best_r = max(res_to_mod.items(), key=lambda x: x[1])[0]
        best_partition = res_to_partition[best_r]
        best_modularity = res_to_mod[best_r]

        communities = {}
        for node, cid in enumerate(best_partition):
            communities.setdefault(cid, []).append(node)

        partition_list = list(communities.values())

        with open(f"comm_res{best_r}.pkl", 'wb') as f:
            pickle.dump(partition_list, f)



    all_community_ids = community_ids_per_res
    print (f"Final resoluton:{resolutions}")
    # Create embedding layers, each with size matching number of communities for that resolution
    community_embeddings = nn.ModuleList([
        nn.Embedding(num_comms, embedding_dim, device=device)
        for num_comms in num_communities_list
    ])
    
    # ---------------------------
    # 🔹 Save results into cache folder
    # ---------------------------
    """         
    if cache_file is not None:
        torch.save({
            "best_r": best_r,
            "community_embeddings": [emb.state_dict() for emb in community_embeddings],
            "all_community_ids": all_community_ids,
            "num_communities_list": num_communities_list,
            "resolutions": resolutions
        }, cache_file)
        print(f"[💾] Saved Louvain embeddings into {cache_file}")
    """


    return best_r, community_embeddings, all_community_ids, num_communities_list, resolutions
