## Environment Setup.

```
# 1) Create & activate
  conda create -y -n pyg_env python=3.10
  conda activate pyg_env
# 2) Install PyTorch 2.4.0 (+ torchvision/torchaudio) with CUDA 12.4 runtime
conda install -y -c pytorch -c nvidia \
  pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4

# 3) Install PyTorch Geometric 2.6.1 and ops (match torch==2.4.0+cu124)
pip install -U torch_geometric==2.6.1 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

pip install -U pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
python -m pip install -U   "rmm-cu12==24.06.*"   "cudf-cu12==24.06.*"   "pylibcugraph-cu12==24.06.*"   "pylibcugraphops-cu12==24.06.*"   "cugraph-cu12==24.06.*"   --extra-index-url https://pypi.nvidia.com



pip install -U --no-cache-dir ogb numpy   scipy   pandas   scikit-learn   networkx   python-louvain   gdown
~                                                                                                              
```

## Run Experiments.

```
sh run.sh
```
