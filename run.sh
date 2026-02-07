python main_community.py --dataset amazon-ratings --res 'None' --comres 1 --epochs 1500 --batch_size 512 --min_q 0.6 --del_q 0.1 --runs 10 --metric acc --display_step 100
python main_community.py --dataset amazon-ratings --res 'None' --comres 1 --epochs 1500 --batch_size 512 --min_q 0.6 --del_q 0.1 --runs 10 --metric acc --display_step 100 --LPF 1

python main_community.py --dataset roman-empire --res 'None' --comres 1 --epochs 600 --batch_size 11600 --min_q 1.0 --del_q 0.1 --runs 10 --metric acc --display_step 100  --lr .01 --hidden_channels 792 --num_layers 3
python main_community.py --dataset roman-empire --res 'None' --comres 1 --epochs 600 --batch_size 11600 --min_q 1.0 --del_q 0.1 --runs 10 --metric acc --display_step 100  --lr .01 --hidden_channels 792 --num_layers 3 --LPF 1 --NF

python main_community.py --dataset cora --res 'none' --comres 1 --epochs 200 --hidden_channels 256  --batch_size 128  --min_q 0.1 --del_q 0.2  --train_prop 0.6 --valid_prop 0.2 --rand_split --runs 10 --display_step 100 --metric acc
python main_community.py --dataset cora --res 'none' --comres 1 --epochs 200 --hidden_channels 256  --batch_size 128  --min_q 0.1 --del_q 0.2  --train_prop 0.6 --valid_prop 0.2 --rand_split --runs 10 --display_step 100 --metric acc --NF --LPF 2

python main_community.py --dataset squirrel-filtered --res '0' --comres 1 --epochs 20 --hidden_channels 600 --batch_size 600 --num_layers 3 --dropout 0.8 --min_q .61 --del_q 0.05 --runs 10 --metric acc --display_step 3 --lr .005 --weight_decay 0
python main_community.py --dataset squirrel-filtered --res '0' --comres 1 --epochs 20 --hidden_channels 600 --batch_size 600 --num_layers 3 --dropout 0.8 --min_q .61 --del_q 0.05 --runs 10 --metric acc --display_step 3 --lr .005 --weight_decay 0 --LPF 2


python main_community.py --dataset chameleon-filtered --res 'none' --comres 1 --epochs 30 --batch_size 256 --num_layers 1 --dropout 0.5 --min_q 0.7 --del_q 0.1 --train_prop 0.6 --valid_prop 0.2 --runs 10 --metric acc --lr .001
python main_community.py --dataset chameleon-filtered --res 'none' --comres 1 --epochs 30 --batch_size 256 --num_layers 1 --dropout 0.5 --min_q 0.7 --del_q 0.1 --train_prop 0.6 --valid_prop 0.2 --runs 10 --metric acc --lr .001 --LPF 2

python main_community.py   --dataset actor   --res 'None' --comres 1   --epochs 400   --batch_size 7600   --num_layers 3   --dropout 0.8   --hidden_channels 256   --lr 0.0001 --weight_decay 0.001   --train_prop 0.6 --valid_prop 0.2   --runs 10 --rand_split   --metric acc --display_step 50 --min_q 1 --del_q 0.1 --weight_decay 0.001
python main_community.py   --dataset actor   --res 'None' --comres 1   --epochs 400   --batch_size 7600   --num_layers 3   --dropout 0.8   --hidden_channels 256   --lr 0.0001 --weight_decay 0.001   --train_prop 0.6 --valid_prop 0.2   --runs 10 --rand_split   --metric acc --display_step 50 --min_q 1 --del_q 0.1 --weight_decay 0.001 --LPF 1

python main_community.py --dataset tolokers --res '0.5 0.75 1 1.364' --comres 1 --epochs 2000 --batch_size 512 --dropout 0.5 --num_layers 2 --min_q 0.3 --del_q 0.1 --metric rocauc --runs 10 --display_step 100
python main_community.py --dataset tolokers --res '0.5 0.75 1 1.364' --comres 1 --epochs 2000 --batch_size 512 --dropout 0.5 --num_layers 2 --min_q 0.3 --del_q 0.1 --metric rocauc --runs 10 --display_step 100 --LPF 2
python main_community.py --dataset tolokers --res '0.5 0.75 1 1.364' --comres 1 --epochs 2000 --batch_size 512 --dropout 0.5 --num_layers 2 --min_q 0.3 --del_q 0.1 --metric rocauc --runs 10 --display_step 100 --LPF 2 --NF

python main_community.py --dataset pubmed --res '0.5 1 1.956' --comres 1 --epochs 300 --batch_size 8000 --dropout 0.7 --num_layers 3 --min_q 0.7 --del_q 0.1 --metric acc --runs 10 --display_step 100  --train_prop 0.6 --valid_prop 0.2 --rand_split --lr .001

python main_community.py --dataset questions --res 'none' --comres 1 --epochs 70 --batch_size 256 --num_layers 0 --dropout 0.8 --min_q 0.5 --del_q 0.1 --train_prop 0.6 --valid_prop 0.2 --runs 10 --metric rocauc --lr .0001 --weight_decay 0 --hidden_channels 128
python main_community.py --dataset questions --res 'none' --comres 1 --epochs 70 --batch_size 256 --num_layers 0 --dropout 0.8 --min_q 0.5 --del_q 0.1 --train_prop 0.6 --valid_prop 0.2 --runs 10 --metric rocauc --lr .0001 --weight_decay 0 --hidden_channels 128 --LPF 2
python main_community.py --dataset questions --res 'none' --comres 1 --epochs 70 --batch_size 256 --num_layers 0 --dropout 0.8 --min_q 0.7 --del_q 0.1 --train_prop 0.6 --valid_prop 0.2 --runs 10 --metric rocauc --lr .0001 --weight_decay 0 --hidden_channels 128 --LPF 2 --NF












python main_community.py --dataset flickr --res '0.5 0.516 0.531 0.562 0.594 0.625 0.656 0.672 0.688 0.719 0.75 0.781 0.812 0.844 0.875 1 1.051 1.076 1.102 1.203 1.304 1.406 1.507 1.558 1.609 1.711 1.812 2.015 2.117 2.218 2.624 2.778 2.932 3.239 3.854 4.162 4.469 5.084 5.699 6.314 6.929 7.544 8.358 9.173 9.988 10.802 12.431 14.06 15.689 17.318 18.946 20.575 23.23 25.884 31.194 36.504 39.159 41.814 52.434 57.744 63.054 72.348 81.643 90.938 100.232 109.526 118.82 137.409 155.998 174.587 193.176 211.765' --num_layer 2 --epochs 150 --batch_size 8192 --runs 10 --min_q 0.1 --dropout .8 --lr .00005 --metric acc --del_q 0.01 --hidden_channel 512 --weight_decay 0.001 --LPF 2 --NF

python main_community.py --dataset reddit2 --res 'None' --epochs 1000 --batch_size 8000 --runs 10 --min_q  0.3 --del_q 0.3 --metric acc
python main_community.py --dataset reddit2 --res '0' --epochs 1550 --batch_size 20000 --runs 10 --min_q 0.3 --del_q 0.3 --metric acc --hidden_channel 1024 --display_step 500 --lr .0007 --weight_decay .0000005 --dropout .6 --LPF 2

python main_community.py --dataset yelp --res 'None' --epochs 300 --batch_size 32000 --hidden_channels 2048 --num_layers 5 --dropout 0.5 --runs 10 --min_q 1 --lr .00005 --LPF 2
python main_community.py --dataset amazon-products --res 'None' --epochs 200 --batch_size 64000 --hidden_channels 2048 --num_layers 5  --dropout 0.5 --runs 10 --min_q 1.0 --eval_batch --lr .00005
python main_community.py --dataset ogbn-products --res '0.5 1 3.214 11.941 30.679 57.611 108.799 200.231 362.663 675.602 1183.541' --epochs 400 --batch_size 100000 --lr .0001 --weight_decay 0.005 --dropout 0.5 --runs 10 --metric acc --min_q 0.3 --del_q 0.2 --hidden_channels 1600 --eval_batch --display_step 50
python main_community.py --dataset ogbn-products --res '0.5 1 3.214 11.941 30.679 57.611 108.799 200.231 362.663 675.602 1183.541' --epochs 600 --batch_size 100000 --lr .0001 --weight_decay 0.005 --dropout 0.6 --runs 10 --metric acc --min_q 0.3 --del_q 0.2 --hidden_channels 792 --eval_batch --display_step 100 --LPF 3
