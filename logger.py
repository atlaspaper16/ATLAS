import torch

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self,args,clustering_time, training_time, run=None,  mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {ind}')
            print(f'Final Train: {result[ind, 0]:.2f}')
            print(f'Final Test: {result[ind, 2]:.2f}')
            self.test=result[ind, 2]
        else:
            result = 100 * torch.tensor(self.results)
            
            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)
            
            print(f'All runs:')
            r0 = best_result[:, 0]
            #print(f'Highest Train: {r0.mean():.2f} ± {r0.std():.2f}')
            r1 = best_result[:, 1]
            #print(f'Highest Test: {r1.mean():.2f} ± {r1.std():.2f}')
            r2 = best_result[:, 2]
            #print(f'Highest Valid: {r2.mean():.2f} ± {r2.std():.2f}')
            r3 = best_result[:, 3]
            #print(f'  Final Train: {r3.mean():.2f} ± {r3.std():.2f}')
            r4 = best_result[:, 4]
            #print(f'   Final Test: {r4.mean():.2f} ± {r4.std():.2f}')
            
            self.test=r.mean()

            if not os.path.exists(f'results'):
                os.makedirs(f'results')
            args.model = "MLP"
            filename = f'results/{args.dataset}.csv'
            print(f"Saving results to {filename}")
            with open(f"{filename}", 'a+') as write_obj:
                write_obj.write(
                        f"{args.dataset}, "
                        f"{args.res}, "
                        f"{args.epochs}, "
                        f"{args.num_layers}, "
                        f"hidden_channels: {args.hidden_channels}, "
                        f"emb_dim: {args.emb_dim}, "
                        f"batch_size: {args.batch_size}, "
                        f"lr: {args.lr}, "
                        f"min_q: {args.min_q}, "
                        f"del_q: {args.del_q}, "
                        f"weight_decay: {args.weight_decay}, "
                        f"dropout: {args.dropout}, "
                        f"seed: {args.seed}, "
                        f"metric: {args.metric}, "
                        f"LPF: {args.LPF}, "
                        f"NF: {args.NF}, "
                        f'Highest Train: {r0.mean():.2f} ± {r0.std():.2f}, '
                        f'Highest Test: {r1.mean():.2f} ± {r1.std():.2f}, '
                        f'Highest Valid: {r2.mean():.2f} ± {r2.std():.2f}, '
                        f'Final Train: {r3.mean():.2f} ± {r3.std():.2f}, '
                        f'Final Test: {r4.mean():.2f} ± {r4.std():.2f}, '
                        f'Clustering time: {clustering_time}, '
                        f'Training time: {training_time}, '
                        f'Total time: {clustering_time + training_time},\n'
                    )




            return best_result[:, 4]

    def output(self,out_path,info):
        with open(out_path,'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')

import os
def save_model(args, model, optimizer, run):
    if not os.path.exists(f'models/{args.dataset}'):
        os.makedirs(f'models/{args.dataset}')
    if(args.model=='MPNN'):
        model_path = f'models/{args.dataset}/{args.model}_{args.gnn}_{run}.pt'
    else:
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path)

def load_model(args, model, optimizer, run):
    if(args.model=='MPNN'):
        model_path = f'models/{args.dataset}/{args.model}_{args.gnn}_{run}.pt'
    else:
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def save_result(args, results):
    if not os.path.exists(f'results'):
        os.makedirs(f'results')
    args.model = "MLP"
    filename = f'results/{args.dataset}.csv'
    print(f"Saving results to {filename}")
