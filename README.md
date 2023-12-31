
# RePHINE  
 Johanna Immonen | [Amauri H. Souza](https://www.amauriholanda.org) |  [Vikas Garg](https://www.mit.edu/~vgarg/)

This is the official repo for the paper [Going beyond persistent homology using persistent homology](https://nips.cc/virtual/2023/oral/73876) (NeurIPS 2023).

In our work, we study the expressive power of PH on attributed (or colored) graphs. In particular, we establish lower and 
upper bounds for the expressivity of 0-dimensional persistence diagrams obtained from vertex- and edge-level filtration functions. 
Based on our insights, we present RePHINE (short for ''**Re**fining **PH** by **I**ncorporating **N**ode-color into **E**dge-based filtration''), 
a simple method that exploits a subtle interplay between vertex- and edge-level persistence information to improve the expressivity of color-based PH. 
We show the effectiveness of RePHINE on three synthetic and six real datasets. 


## Requirements

If you use conda as package manager you can use to create a virtual environment:
```bash
  conda create -n rephine 
  conda activate rephine
```
or if use venv
```
  python -m venv env 
  source env/bin/activate         #If you use Linux or Mac
  .\venv\Scripts\Activate.ps1     #If you use Windows
```
you can install dependencies in ```requirements.txt```

```
  pip install -r requirements.txt

```
If you prefer to install manually

```
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
  pip install ogb
  pip install dgl
  pip install networkx
  pip install h5py
```
After that, you have to install the necessary dependencies that computes persistence diagrams

```
  cd torch_ph
  python setup_rephine.py install 
  python setup_std_ph.py install
```

## Usage

### Synthetic
For the experiments with synthetic data, we first create the datasets:
```
python datasets/create_data.py --dataset cub08 --num_alien_nodes 1
python datasets/create_data.py --dataset cub10 --num_alien_nodes 2
python datasets/create_data.py --dataset cub12 --num_alien_nodes 3
```
and then run the scripts for rephine, standard (PH), and GCN:
```
python main_toy.py --setting cub10-2  --seed 8 --model rephine --reduce_tuples --no-dim1
python main_toy.py --setting cub10-2  --seed 8 --model standard --no-dim1
python main_toy.py --setting cub10-2  --seed 8 --model gcn 
```

### Main experiments

For the main experiments, we run the ```main.py``` with the arguments in ```cli_main.py```. For instance, to run RePHINE
and Standard on NCI109 combined with a GCN model, we run:
```
python main.py --dataset NCI109  --diagram_type rephine --gnn gcn   
python main.py --dataset NCI109  --diagram_type standard --gnn gcn   
```

### Comparison to PersLay

We first need to generate data using the file ```datasets/generate_data_perslay.py```. Then, to run RePHINE+Linear, we choose 
```--gnn linear``` and run
```
python main.py --dataset NCI109  --diagram_type rephine --gnn linear   
```

To run the PersLay model, we use the file ```main_perslay.py```, e.g.,
```
python main_perslay.py --dataset NCI109    
```

## Citation
```
@inproceedings{rephine,
  title={Going beyond persistent homology using persistent homology},
  author={Johanna Immonen and Amauri H. Souza and Vikas Garg},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Acknowledgments
Some of the routines in this repo were modified from [TOGL](https://github.com/BorgwardtLab/TOGL) and [PersLay](https://github.com/MathieuCarriere/perslay).

