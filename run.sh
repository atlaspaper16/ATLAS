python main_community.py --dataset amazon-ratings --res 'None' --comres 1 --epochs 1500 --batch_size 512 --min_q 0.6 --del_q 0.1 --runs 10 --metric acc --display_step 100
python main_community.py --dataset roman-empire --res 'None' --comres 1 --epochs 500 --batch_size 512 --min_q 1.0 --del_q 0.1 --runs 10 --metric acc
python main_community.py --dataset cora --res 'none' --comres 1 --epochs 200 --hidden_channels 256  --batch_size 128  --min_q 0.1 --del_q 0.2  --train_prop 0.6 --valid_prop 0.2 --rand_split --runs 10 --display_step 100 --metric acc

python main_community.py --dataset squirrel-filtered --res '.1' --comres 1 --epochs 40 --batch_size 512 --num_layers 3 --dropout .5 --min_q .61 --del_q 0.05 --runs 10 --metric acc --display_step 20 --lr .005


python main_community.py --dataset chameleon-filtered --res 'none' --comres 1 --epochs 30 --batch_size 256 --num_layers 1 --dropout 0.5 --min_q 0.7 --del_q 0.1 --train_prop 0.6 --valid_prop 0.2 --runs 10 --metric acc --lr .001
python main_community.py --dataset actor --res 'None' --comres 1 --epochs 200 --batch_size 128 --num_layers 3  --dropout 0.8 --min_q 1 --del_q 0.1 --train_prop 0.6 --valid_prop 0.2 --runs 10 --rand_split --metric acc
python main_community.py --dataset tolokers --res '0.5 0.75 1 1.364' --comres 1 --epochs 2000 --batch_size 512 --dropout 0.5 --num_layers 2 --min_q 0.3 --del_q 0.1 --metric rocauc --runs 10 --display_step 100
python main_community.py --dataset pubmed --res '0.5 1 1.956' --comres 1 --epochs 300 --batch_size 8000 --dropout 0.7 --num_layers 3 --min_q 0.7 --del_q 0.1 --metric acc --runs 10 --display_step 100  --train_prop 0.6 --valid_prop 0.2 --rand_split --lr .001















python main_community.py --dataset flickr --res 'None' --num_layer 2 --epochs 20 --batch_size 1024 --runs 10 --min_q 0.1 --dropout .7 --lr .001 --metric acc --del_q 0.01 --hidden_channel 256
python main_community.py --dataset reddit2 --res 'None' --epochs 1000 --batch_size 8000 --runs 10 --min_q  0.3 --del_q 0.3 --metric acc
python main_community.py --dataset yelp --res 'None' --epochs 300 --batch_size 32000 --hidden_channels 2048 --num_layers 5 --dropout 0.5 --runs 10 --min_q 1 --lr .00005
python main_community.py --dataset amazon-products --res 'None' --epochs 200 --batch_size 64000 --hidden_channels 2048 --num_layers 5  --dropout 0.5 --runs 10 --min_q 1.0 --eval_batch --lr .00005
python main_community.py --dataset ogbn-products --res 'None' --epochs 400 --batch_size 32000 --dropout 0.5 --runs 10 --metric acc --min_q 0.3 --del_q 0.1
