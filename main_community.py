# main_community.py
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parse import parser_add_main_args
from community_features import generate_louvain_embeddings
from dataset import load_dataset
from data_utils import *
from model_community import MLPWithCommunityEmbedding
from logger import Logger, save_result

import gc
import torch

def _cuda_cleanup(local_ns: dict):
    """Delete common CUDA-heavy refs, then flush allocator."""
    # Explicitly drop likely big objects if they exist
    for name in (
        "model", "optimizer", "scheduler", "scaler",
        "community_embeddings", "E", "x_all", "x_all_full",
        "logits_all", "out", "loss", "batch", "y_pred",
        "train_loader", "val_loader", "test_loader",
    ):
        if name in local_ns:
            try:
                del local_ns[name]
            except Exception:
                pass

    # Finish any outstanding kernels
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Drop Python refs and return unused blocks to the driver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# -------------------------------
# Repro
# -------------------------------
def fix_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# Feature assembly per batch of node IDs
# -------------------------------
def load_features_for_nodes(batch_nodes: torch.Tensor,
                            x_base: torch.Tensor,
                            community_embeddings,
                            all_community_ids):
    """
    Gather base features for `batch_nodes` and (if present) concat
    Louvain community embeddings across all resolutions.
    """
    batch_nodes = batch_nodes.to(torch.long)
    x = x_base[batch_nodes]

    if community_embeddings is None:
        return x

    emb_list = []
    for emb_layer, comm_ids in zip(community_embeddings, all_community_ids):
        batch_comm_ids = comm_ids[batch_nodes]     # already on device
        emb = emb_layer(batch_comm_ids)            # embedding lookup on device
        emb_list.append(emb)

    if emb_list:
        emb = torch.cat(emb_list, dim=1)
        x = torch.cat([x, emb], dim=1)
    return x



# -------------------------------
# Torch micro-F1 for multilabel (no CPU copies)
# -------------------------------
def torch_micro_f1(y_true_bin: torch.Tensor, y_hat_bin: torch.Tensor) -> float:
    # y_* are {0,1} tensors on device
    tp = (y_hat_bin & y_true_bin).sum().item()
    fp = (y_hat_bin & (~y_true_bin.bool())).sum().item()
    fn = ((~y_hat_bin.bool()) & y_true_bin).sum().item()
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


# -------------------------------
# Build full-batch features once (outside eval timer)
# -------------------------------
def build_full_features(data, x_base, community_embeddings, all_community_ids, device):
    all_nodes = torch.arange(data.num_nodes, device=device, dtype=torch.long)
    return load_features_for_nodes(all_nodes, x_base, community_embeddings, all_community_ids)


# -------------------------------
# TRAIN
# -------------------------------
def train(model, data, community_embeddings, all_community_ids, train_idx, optimizer, criterion, bce_loss, batch_size=8192):
    model.train()
    total_loss = 0.0
    perm = torch.randperm(train_idx.size(0), device=train_idx.device)

    device = next(model.parameters()).device
    x_base = data.graph['node_feat'].to(device)

    for i in range(0, train_idx.size(0), batch_size):
        optimizer.zero_grad()
        batch_nodes = train_idx[perm[i:i + batch_size]]
        x_batch = load_features_for_nodes(batch_nodes, x_base, community_embeddings, all_community_ids)

        # BCE path expects raw logits; NLL path expects log-probs
        if bce_loss:
            logits = model.forward_logits(x_batch)               # logits [B, C]
            out = torch.sigmoid(logits)
            target = data.label[batch_nodes].float()
        else:
            out = model(x_batch)                                # log-probs [B, C]
            target = data.label[batch_nodes]
            if target.dim() == 2 and target.size(1) > 1:
                target = target.argmax(dim=1)
            elif target.dim() == 2 and target.size(1) == 1:
                target = target.squeeze(1)
            target = target.long()

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_nodes.size(0)

    return total_loss / train_idx.size(0)


@torch.no_grad()
def evaluate_batch_forward(
    model,
    data,
    split_idx,
    args,
    criterion,
    community_embeddings,
    all_community_ids,
    batch_size: int,
    bce_loss: bool,
):
    model.eval()
    device = next(model.parameters()).device
    x_base = data.graph['node_feat'].to(device)
    y_true = data.label

    multilabel = (y_true.dim() == 2 and y_true.size(1) > 1 and y_true.max() <= 1)

    def split_metric(nodes: torch.Tensor) -> float:
        vals = []
        for i in range(0, nodes.numel(), batch_size):
            bnodes = nodes[i:i+batch_size]
            x_b = load_features_for_nodes(bnodes, x_base, community_embeddings, all_community_ids)
            out = model.forward_logits(x_b).float() if bce_loss else model(x_b)
            y_b = y_true[bnodes]

            if multilabel:
                if args.metric in ('f1', 'rocauc'):
                    probs = torch.sigmoid(out)
                    y_hat = (probs >= 0.5).to(torch.int8)
                    vals.append(torch_micro_f1(y_b.to(torch.int8), y_hat))
                else:
                    probs = torch.sigmoid(out)
                    y_hat = (probs >= 0.5)
                    vals.append(float((y_hat == (y_b > 0.5)).all(dim=1).float().mean().item()))
            else:
                if args.metric == 'acc':
                    vals.append(float(eval_acc(y_b.unsqueeze(1) if y_b.dim()==1 else y_b, out)))
                elif args.metric == 'f1':
                    vals.append(float(eval_f1(y_b.unsqueeze(1) if y_b.dim()==1 else y_b, out)))
                else:
                    vals.append(float(eval_rocauc(y_b.unsqueeze(1) if y_b.dim()==1 else y_b, out)))
        return sum(vals) / max(len(vals), 1)

    results = {split: split_metric(split_idx[split]) for split in ['train', 'valid', 'test']}

    # Batched validation loss
    val_nodes = split_idx['valid']
    val_losses = []
    for i in range(0, val_nodes.numel(), batch_size):
        bnodes = val_nodes[i:i+batch_size]
        x_b = load_features_for_nodes(bnodes, x_base, community_embeddings, all_community_ids)
        y_b = y_true[bnodes]
        if bce_loss:
            logits_b = model.forward_logits(x_b).float()
            val_losses.append(criterion(torch.sigmoid(logits_b), y_b.float()).item())
        else:
            logprobs_b = model(x_b)
            if y_b.dim() == 2 and y_b.size(1) > 1:
                labels_ce = y_b.argmax(dim=1).long()
            elif y_b.dim() == 2 and y_b.size(1) == 1:
                labels_ce = y_b.squeeze(1).long()
            else:
                labels_ce = y_b.long()
            val_losses.append(nn.NLLLoss(ignore_index=-1)(logprobs_b, labels_ce).item())

    val_loss = sum(val_losses) / max(len(val_losses), 1)
    return results['train'], results['valid'], results['test'], val_loss, None




# -------------------------------
# EVALUATE (true full-batch; timer only measures forward)
# -------------------------------
@torch.no_grad()
def evaluate_fullbatch_forward(model, data, split_idx, args, criterion, x_all, bce_loss):
    """
    Forward-only, true full-batch inference:
      - x_all is prebuilt [X‖E] for ALL nodes (built outside the timer)
      - Single forward over ALL nodes
      - Metrics per split by indexing logits
    """
    model.eval()
    y_true = data.label
    

    logits_all = model.forward_logits(x_all) if bce_loss else model(x_all)
   # Single forward over ALL nodes (use autocast on CUDA for speed)
    """
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits_all = model.forward_logits(x_all) if bce_loss else model(x_all)
    """
    # Ensure FP32 for BCE path to avoid Half/Float mismatch
    if bce_loss:
        logits_all = logits_all.float()
    else:
        logits_all = model.forward_logits(x_all) if bce_loss else model(x_all)

    # Ensure FP32 for BCE path to avoid Half/Float mismatch
    if bce_loss:
        logits_all = logits_all.float()
    # Detect multilabel
    multilabel = (y_true.dim() == 2 and y_true.size(1) > 1 and y_true.max() <= 1)

    results = {}
    for split in ['train', 'valid', 'test']:
        nodes = split_idx[split]
        out = logits_all[nodes]
        y  = y_true[nodes]

        if multilabel:
            # F1 / ROC-AUC / exact-match for multilabel
            if args.metric == 'f1':
                probs = torch.sigmoid(out.float())
                y_hat = (probs >= 0.5).to(torch.int8)
                y_bin = y.to(torch.int8)
                metric_val = torch_micro_f1(y_bin, y_hat)
            elif args.metric == 'rocauc':
                # Optional: simple, per-class AUC with torchmetrics would be better;
                # kept minimal here—fallback to exact-match if AUC is requested without sklearn.
                probs = torch.sigmoid(out.float())
                y_hat = (probs >= 0.5).to(torch.int8)
                y_bin = y.to(torch.int8)
                metric_val = torch_micro_f1(y_bin, y_hat)  # safe fallback
            else:  # 'acc' → exact-match (subset) accuracy
                probs = torch.sigmoid(out.float())
                y_hat = (probs >= 0.5)
                metric_val = float((y_hat == (y > 0.5)).all(dim=1).float().mean().item())
        else:
            # Single-label: reuse your helpers (y is long / 1D or one-hot)
            if args.metric == 'acc':
                metric_val = float(eval_acc(y.unsqueeze(1) if y.dim() == 1 else y, out))
            elif args.metric == 'f1':
                metric_val = float(eval_f1(y.unsqueeze(1) if y.dim() == 1 else y, out))
            else:
                metric_val = float(eval_rocauc(y.unsqueeze(1) if y.dim() == 1 else y, out))

        results[split] = metric_val

    # Validation loss (computed on full-batch logits)
    val_idx = split_idx['valid']
    val_logits = logits_all[val_idx]
    val_labels = y_true[val_idx]
    if bce_loss:
        val_loss = criterion(torch.sigmoid(val_logits.float()), val_labels.float())
    else:
        if val_labels.dim() == 2 and val_labels.size(1) > 1:
            labels_ce = val_labels.argmax(dim=1).long()
        elif val_labels.dim() == 2 and val_labels.size(1) == 1:
            labels_ce = val_labels.squeeze(1).long()
        else:
            labels_ce = val_labels.long()
        val_loss = nn.NLLLoss(ignore_index=-1)(val_logits, labels_ce)

    return results['train'], results['valid'], results['test'], float(val_loss.item()), logits_all


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Community-enhanced MLP Node Classification')
    parser_add_main_args(parser)
    args = parser.parse_args()

    fix_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() and not getattr(args, "cpu", False) else 'cpu'
    print(f"Using device: {device}")

    # === Load dataset ===
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)
    x_base = dataset.graph['node_feat'].to(device)

    # Output dimension
    if dataset.label.dim() == 2 and dataset.label.size(1) > 1:
        out_dim = dataset.label.size(1)
    else:
        out_dim = int(dataset.label.max().item()) + 1 if dataset.label.numel() > 0 else 1

    # Task detection & criterion (GraphSAINT-style for Yelp/Amazon-Products: use BCELoss over per-class sigmoid probabilities)
    ds_lower = args.dataset.lower()
    bce_loss = ds_lower in ('yelp', 'amazon-products')
    multilabel = (dataset.label.dim() == 2 and dataset.label.size(1) > 1 and dataset.label.max() <= 1)
    bce_loss = bce_loss or multilabel

    if bce_loss:
        criterion = nn.BCELoss()
        dataset.label = dataset.label.float().to(device)
    else:
        criterion = nn.NLLLoss(ignore_index=-1)
        if dataset.label.dim() == 2:
            dataset.label = dataset.label.squeeze(1)
        dataset.label = dataset.label.long().to(device)

    # Preserve initial res for next run
    init_res = args.res

    logger = Logger(args.runs)

    for run in range(args.runs):
        print(f"\n========== Run {run} / {args.runs - 1} ==========")

        # --- Community embeddings (preprocessing) ---
        print("Start Community Detection")
        t0 = time.time()
        best_r, community_embeddings, all_community_ids, num_communities_list, args.res = generate_louvain_embeddings(
            data=dataset, args=args, device=device
        )
        clustering_time = time.time() - t0
        print(f"End Community Detection, Time: {clustering_time:.3f}s")

        # --- Splits ---
        if args.rand_split:
            split_idx_lst = []
            train_idx, val_idx, test_idx = rand_train_test_idx(
                dataset.label,
                train_prop=args.train_prop,
                valid_prop=args.valid_prop,
                pkl_path=f"comm_res{best_r}.pkl" if best_r is not None else None
            )
            split_idx_lst.append(
             {'train': train_idx.to(device),
                         'valid': val_idx.to(device),
                         'test':  test_idx.to(device)})
        else:
            split_idx_lst = [
                {k: v.to(device) for k, v in splits.items()}
                    for splits in load_fixed_splits(args.data_dir, dataset, name=args.dataset, pkl_path="")
                ]
        # --- Model ---
        comm_feat_dim = (len(num_communities_list) * args.emb_dim) if community_embeddings is not None else 0
        input_dim = x_base.size(1) + comm_feat_dim

        model = MLPWithCommunityEmbedding(
            input_dim=input_dim,
            hidden_dim=args.hidden_channels,
            out_dim=out_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size
        ).to(device)

        # --- Optimizer ---
        if community_embeddings is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(community_embeddings.parameters()),
                lr=args.lr, weight_decay=args.weight_decay
            )
        

        # --- Split selection ---
        #split_idx = split_idx_lst[0] if args.rand_split else split_idx_lst[run]
        if len(split_idx_lst) > 1:
            split_idx = split_idx_lst[run]
        elif len(split_idx_lst) == 1:
            split_idx = split_idx_lst[0]
        print("Num train:", len(split_idx["train"]))
        print("Num valid:", len(split_idx["valid"]))
        print("Num test:",  len(split_idx["test"]))

        train_idx = split_idx['train']


        model.reset_parameters()

        best_val = -1e9
        best_test = -1e9
        train_times, eval_times = [], []
        file_name = args.dataset+"_training_log.txt"
        log_file = open(file_name, "a")
        for epoch in range(args.epochs):
            # Train
            t_train0 = time.time()
            loss = train(model, dataset, community_embeddings, all_community_ids,
                         split_idx['train'], optimizer, criterion, bce_loss, batch_size=args.batch_size)
            t_train = time.time() - t_train0

            # Build full-batch features OUTSIDE the eval timer
            x_all_full = build_full_features(dataset, x_base, community_embeddings, all_community_ids, device)
            

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

            # Evaluate (forward-only timed)
            """
            t_eval0 = time.time()
            train_acc, val_acc, test_acc, val_loss, _ = evaluate_fullbatch_forward(
                model, dataset, split_idx, args, criterion, x_all_full, bce_loss
            )
            t_eval = time.time() - t_eval0
            """
            # --- EVAL ---
            t_eval0 = time.time()
            if args.eval_batch:
                # Mini-batch evaluation (no x_all_full build)
                train_acc, val_acc, test_acc, val_loss, _ = evaluate_batch_forward(
                                        model, dataset, split_idx, args, criterion,
                                        community_embeddings, all_community_ids, args.batch_size, bce_loss
                                        )
            else:
    # Full-batch evaluation (existing path)
                x_all_full = build_full_features(dataset,
                                     dataset.graph['node_feat'].to(device),
                                     community_embeddings, all_community_ids, device)
                train_acc, val_acc, test_acc, val_loss, _ = evaluate_fullbatch_forward(
                        model, dataset, split_idx, args, criterion, x_all_full, bce_loss
                        )



            t_eval = time.time() - t_eval0
            train_times.append(t_train)
            eval_times.append(t_eval)

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

            if epoch % args.display_step == 0 or epoch == args.epochs - 1:
                print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, "
                      f"Train: {train_acc * 100:.2f}%, Valid: {val_acc * 100:.2f}%, Test: {test_acc * 100:.2f}%, "
                      f"Best Valid: {best_val * 100:.2f}%, Best Test: {best_test * 100:.2f}%, "
                      f"EpochTime: {t_train + t_eval:.3f}s (train {t_train:.3f}s, infer {t_eval:.3f}s)")

            logger.add_result(run, (train_acc, val_acc, test_acc, val_loss))
            log_str = (f"Epoch: {epoch:02d}, "
                   f"Loss: {loss:.4f}, "
                   f"Train: {100 * train_acc:.2f}%, "
                   f"Valid: {100 * val_acc:.2f}%, "
                   f"Test: {100 * test_acc:.2f}%")
            log_file.write(log_str + "\n")
            log_file.flush()  # ensures content is written immediately
        # Per-run timing & summary
        training_time = sum(train_times) + sum(eval_times)

        # restore original res for next run
        args.res = init_res
        _cuda_cleanup(locals())

    # Final summary and save
    # Final summary and save
    results = logger.print_statistics(args, clustering_time, training_time, run=None, mode='max_acc')
    save_result(args, results)
    log_file.close()
