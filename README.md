# modalnext
Unified heterogeneous cellular data integration

# Dataset
We have respectively collected the corresponding datasets for benchmarking according to the four summarized integration tasks. We will introduce the datasets for each task. All datasets were converted by us into the "anndata" format supported by the deep learning framework.

Download: https://drive.google.com/drive/folders/1Bq2sWXROlEw405NpFW2cpSRgnppf47R5?hl=zh

# Installation
Python version: python 3.8

Install the following dependencies via pip or conda:
```
pip install muon
pip install torchbiggraph
pip install scanpy
pip install pytorch-ignite
pip install typing_extensions
pip install tensorboardX
pip install torch-1.11.0+cu113
```

# Example: MouseSkin multi-modal integration
Enter the example_ma2020 directory and run:

```
python 1_train_inmf.py
python 2_train_gnn_inmf_init.py
```
