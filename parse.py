


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='actor')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=str, default='cuda',
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=0.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=0.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='f1', choices=['acc', 'rocauc','f1'],
                        help='evaluation metric')
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4192)
    parser.add_argument('--min_q', type=float, default=.3)
    parser.add_argument('--del_q', type=float, default=0.1)
    parser.add_argument('--eval_batch', action='store_true',
                    help='Use batched evaluation (mini-batch) instead of full-batch.')

    parser.add_argument('--NF', action='store_true',
                        help='Enable Neighbour Feature (default: disabled)')
    parser.add_argument('--LPF', type=int, default=-1,
                     help='LPF level (int). Enabled if > 0; set 0/-1 to disable.')
    # training
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)# 5e-4
    parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')

    parser.add_argument('--res', type=str, default='1',
                    help="List of Louvain resolutions, e.g. --res 1.0 2.0 3.0 or 'None'")
