# GATI-Rec--Graph Attention Network based Inductive Recommender System
This is an implementation of GATI-Rec.
## Requirements
A full list of dependencies are in the *requirements.txt*:
```
h5py==2.10.0
matplotlib==3.3.3
networkx==2.5
numpy==1.19.2
pandas==1.1.5
PyYAML==6.0
scipy==1.5.2
tensorboardX==2.4.1
torch==1.6.0
torch_geometric==1.6.3
torch_sparse==0.6.8
tqdm==4.55.1
```
Please consult the official installation tutorial if you experience any difficulties.
## Data Preparation
You can train the model smoothly without worrying about downloading datasets since datasets are stored in the *raw_data* folder or will be downloaded automatically.

## Train
To train on MovieLens-100K, type:
```
python train.py
```
or:
```
python train.py --dataset ml_100k
```
To train on MovieLens-1M, Flixster, Douban and YahooMusic, please type these commands respectively:
```
python train.py --dataset ml_1m
python train.py --dataset flixster
python train.py --dataset douban
python train.py --dataset yahoo_music
```
