import os
import sys
sys.path.append("..")
import src.graph_src as si
import anndata
import matplotlib.pyplot as plt
import time 

import scanpy as sc
import numpy as np
import warnings
import torch
from src.spatialknn.preprocess import construct_graph_by_coordinate
from scipy.sparse import coo_matrix
warnings.filterwarnings("ignore")

adata_CG = sc.read("../dataset_repo/spatial/Mouse_Thymus/adata_RNA.h5ad")
adata_CPro = sc.read("../dataset_repo/spatial/Mouse_Thymus/adata_ADT.h5ad")

adata_CG.var_names_make_unique()
adata_CPro.var_names_make_unique()

print(adata_CG.X.max())
print(adata_CPro.X.max())

k_number = 3
workdir = f"../dataset_repo/spatial/Mouse_Thymus/gnn_result_{k_number}nn"
si.settings.set_workdir(workdir)

# ADT预处理
si.pp.filter_genes(adata_CPro,min_n_cells=3)
si.pp.cal_qc_rna(adata_CPro)
si.pp.normalize(adata_CPro,method='lib_size')
si.pp.log_transform(adata_CPro)

si.tl.discretize(adata_CPro,n_bins=5)
si.pl.discretize(adata_CPro,kde=False)
plt.savefig(f"{workdir}/discretize_adt.png")

# RNA预处理
si.pp.filter_genes(adata_CG,min_n_cells=3)
si.pp.cal_qc_rna(adata_CG)
si.pp.normalize(adata_CG,method='lib_size')
si.pp.log_transform(adata_CG)

si.tl.discretize(adata_CG,n_bins=5)
si.pl.discretize(adata_CG,kde=False)
plt.savefig(f"{workdir}/discretize_rna.png")

# 构建空间图关联
cell_position_omics1 = adata_CG.obsm['spatial']
adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=k_number)
# print(adj_omics1)
sparse_matrix = coo_matrix((adj_omics1['value'],(adj_omics1['x'],adj_omics1['y'])), shape = (len(adata_CG.obs_names), len(adata_CG.obs_names)))
# print(sparse_matrix)
CC_adata = sc.AnnData(X=sparse_matrix)
CC_adata.obs_names = adata_CG.obs_names
CC_adata.var_names = adata_CG.obs_names
CC_adata.layers['conn'] = sparse_matrix.toarray()
# print(CC_adata.layers['conn'].nonzero())

# 生成图
si.tl.gen_graph(list_CPro=[adata_CPro],
                list_CG=[adata_CG],
                list_CC=[CC_adata],
                copy=False,
                use_highly_variable=False,
                use_top_pcs=False,
                dirname=f'graph_spatial')

if __name__ == '__main__':
    si.tl.pbg_train(auto_wd=True, save_wd=True, output='model')
    si.pl.pbg_metrics(fig_ncol=3)
    plt.savefig(f"{workdir}/pbg_metrics.png")