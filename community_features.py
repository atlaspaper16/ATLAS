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
# ========== GPU UTILS ==========
import collections
has_gpu = False
try:

    has_gpu = True
except ImportError:
    pass



def num_connected_components(G):
    # cuGraph API moved in newer versions; try new then old
    try:
        df = cugraph.components.connected_components(G)
    except AttributeError:
        df = cugraph.connected_components(G)

    # Column name is usually "labels" (sometimes "component")
    label_col = 'labels' if 'labels' in df.columns else ('component' if 'component' in df.columns else None)
    if label_col is None:
        raise RuntimeError(f"Unexpected CC output columns: {df.columns}")
    return int(df[label_col].nunique()), df  # (count, per-vertex labels)

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
    data, device, min_modularity=0.2, 
     max_modularity_gap=0.1
):
    delta_range=(0.1, .2)
    num_nodes = data.graph['num_nodes']
    tried, partitions = {}, {}
    
    initial_res = [0.5,1]
    print("[🔧] Building graph")

    # --- Run initial resolutions ---
    print("[🚀] Running Louvain on initial resolutions")
    for r in initial_res:
        if device == 'cuda':
            G = build_cugraph(data)
            partition, Q = run_louvain_gpu(G, num_nodes, r)
            #partition, Q = cugraph.leiden(G, random_state=42, resolution=r)
        else:
            partition, Q = run_louvain_cpu(data, num_nodes, r)
        tried[r] = Q
        partitions[r] = partition
        print(f"[✓] Resolution={r} → {max(partition)+1} communities (Q={Q:.4f})")
    # --- Adaptive search loop ---
    while True:
        res_list = sorted(tried.keys())
        mods = [tried[r] for r in res_list]

        # stop if last modularity is too low
        if mods[-1] <= min_modularity and res_list[-1] > 1:
            print(f"[⛔] Modularity {mods[-1]:.4f} ≤ threshold {min_modularity}")
            break

        new_res = None

        # 1. Interpolation: check gaps between known resolutions
        for (r1, r2) in zip(res_list[:-1], res_list[1:]):
            Q1, Q2 = tried[r1], tried[r2]
            delta_q = abs(Q2 - Q1)
            if delta_q > max_modularity_gap:
                new_res = round((r1 + r2) / 2, 3)
                print(f"[➕] Interpolating {r1:.3f}–{r2:.3f} → {new_res} (ΔQ={delta_q:.4f})")
                break

        # 2. Extrapolation: extend beyond max resolution if modularity still good
        if new_res is None:
            max_res, max_Q = res_list[-1], mods[-1]
            if max_Q > min_modularity:
                delta_mod = np.random.uniform(*delta_range)
                Q_target = max_Q - delta_mod
                slope = estimate_slope(tried) or -0.05
                new_res = round(max_res + (Q_target - max_Q) / slope, 3)
                print(f"[🔍] Extrapolating beyond {max_res} → {new_res} targeting Q≈{Q_target:.4f}")

        # 3. Stop if no new resolution found
        if new_res is None or new_res in tried:
            break

        # --- Run Louvain for new resolution ---
        if device == 'cuda':
           # G = build_cugraph(data)
            partition, Q = run_louvain_gpu(G, num_nodes, new_res)
            #partition, Q = cugraph.leiden(G, random_state=42, resolution=new_res)
        else:
            partition, Q = run_louvain_cpu(data, num_nodes, new_res)
        
        print(f"[✓] Resolution={new_res} → {max(partition)+1} communities (Q={Q:.4f})")
        #print(f"Res = {new_res} ->   Q ={Q:.4f}")
        tried[new_res] = Q
        partitions[new_res] = partition

    # --- Cleanup GPU ---
    if device == 'cuda':
        del G
        cleanup_gpu_memory()

    # --- Filter valid resolutions ---
    filtered_resolutions, partition_list, num_communities_list = [], [], []
    filtered_modularity = []
    modularity_dict = {}
    for r in sorted(tried.keys()):
        Q = tried[r]
        if Q >= min_modularity:
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

    # --- Choose best resolution ---
    #selected_initial = [r for r in initial_res if r in filtered_resolutions]
    selected_initial = [r  for r in filtered_resolutions if r >= 1  and r < 10]
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
    diffs = [abs(filtered_modularity[i] - filtered_modularity[i+1]) for i in range(len(filtered_modularity)-1)]
    avg_gap = sum(diffs) / len(diffs)
    print(f"[✅] Used modularity {len(filtered_modularity)} Average modularity gap:{avg_gap}")
    return best_r, filtered_resolutions, partition_list, num_communities_list










def generate_louvain_embeddings(data, args, device='cuda'):
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

        print(f"[🏆] Best resolution = {best_r} → {len(communities)} communities (modularity={best_modularity:.4f})")


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

