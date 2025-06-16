import os
import sys
sys.path.append("..")
import src.graph_src as si
import anndata
import matplotlib.pyplot as plt
import time 

import src.nmf as nmf
import muon as mu
import scanpy as sc
import numpy as np
import warnings
import torch
import scipy.sparse as sp
warnings.filterwarnings("ignore")

def find_indexes(list1, list2):
    indexes = []
    for item in list2:
        index = list1.index(item)
        indexes.append(index)
    return indexes

def generate_gaussian_vectors(lst, length, mean=0, std_dev=1):
    gaussian_vectors = []
    for _ in lst:
        vector = np.random.normal(mean, std_dev, size=(length,))
        gaussian_vectors.append(vector)
    return np.array(gaussian_vectors)

def geneformer_norm(adata, target_sum=10_000):
    # geneformer的预处理方法
    X = adata.X.toarray()
    n_counts = adata.obs['n_counts'].values[:, None]
    median_values = np.array([np.median(X[X[:, i] != 0, i]) for i in range(X.shape[1])])
    X_norm = X / n_counts * target_sum / median_values
    adata.X = sp.csr_matrix(X_norm)

mdata = mu.read_h5mu("../dataset_repo/paired/ma2020/ma2020_preprocessed.h5mu.gz")
print(mdata)

adata_CP = mdata['atac']
adata_CG = mdata['rna']

print(adata_CG.X.max())
print(adata_CP.X.max())
print(adata_CG.X.min())
print(adata_CP.X.min())

workdir = f"../dataset_repo/paired/ma2020/gnn_result_processed"
si.settings.set_workdir(workdir)

si.pp.cal_qc_rna(adata_CP)
geneformer_norm(adata_CP)
si.pp.normalize(adata_CP,method='lib_size')
si.pp.log_transform(adata_CP)

si.tl.discretize(adata_CP,n_bins=5)
si.pl.discretize(adata_CP,kde=False)
plt.savefig(f"{workdir}/discretize_atac.png")

si.pp.cal_qc_rna(adata_CG)
geneformer_norm(adata_CG)
si.pp.normalize(adata_CG,method='lib_size')
si.pp.log_transform(adata_CG)

si.tl.discretize(adata_CG,n_bins=5)
si.pl.discretize(adata_CG,kde=False)
plt.savefig(f"{workdir}/discretize_rna.png")

print(adata_CG.X.max())
print(adata_CP.X.max())

pre_embeddings = {}
mdata = mu.read_h5mu("../dataset_repo/paired/ma2020/inmf/nmf.h5mu.gz")
cells_dict = {name: row for name, row in zip(mdata["rna"].obs_names, mdata.obsm["W_OT"])}
rna_hvg_names = mdata["rna"].var_names[mdata["rna"].var.highly_variable]
rna_hvg_dict = {name: row for name, row in zip(rna_hvg_names, mdata["rna"].uns["H_OT"])}
atac_hvg_names = mdata["atac"].var_names[mdata["atac"].var.highly_variable]
atac_hvg_dict = {name: row for name, row in zip(atac_hvg_names, mdata["atac"].uns["H_OT"])}

gaussian_vectors = generate_gaussian_vectors(adata_CG.obs_names.tolist(), mdata.obsm["W_OT"].shape[1])
for idx, value in enumerate(find_indexes(adata_CG.obs_names.tolist(),mdata["rna"].obs_names.tolist())):
    gaussian_vectors[value] = mdata.obsm["W_OT"][idx]
print(gaussian_vectors.shape)
pre_embeddings[('C', 0)] = torch.tensor(gaussian_vectors, requires_grad=True)

gaussian_vectors = generate_gaussian_vectors(adata_CG.var_names.tolist(), mdata.obsm["W_OT"].shape[1])
for idx, value in enumerate(find_indexes(adata_CG.var_names.tolist(), rna_hvg_names)):
    gaussian_vectors[value] = mdata["rna"].uns["H_OT"][idx]
print(gaussian_vectors.shape)
pre_embeddings[('G', 0)] = torch.tensor(gaussian_vectors, requires_grad=True)

gaussian_vectors = generate_gaussian_vectors(adata_CP.var_names.tolist(), mdata.obsm["W_OT"].shape[1])
for idx, value in enumerate(find_indexes(adata_CP.var_names.tolist(), atac_hvg_names)):
    gaussian_vectors[value] = mdata["atac"].uns["H_OT"][idx]
print(gaussian_vectors.shape)
pre_embeddings[('P', 0)] = torch.tensor(gaussian_vectors, requires_grad=True)

print(pre_embeddings)

start_time = time.time()
si.tl.gen_graph(list_CP=[adata_CP],
                list_CG=[adata_CG],
                copy=False,
                use_highly_variable=False,
                use_top_pcs=False,
                dirname='graph0')
end_time = time.time()
run_time = end_time - start_time
print(f"run time: {run_time}s")

if __name__ == '__main__':
    start_time = time.time()
    si.tl.pbg_train(auto_wd=True, 
                    save_wd=True, 
                    output='model', 
                    pre_embeddings=pre_embeddings
                    )
    end_time = time.time()
    run_time = end_time - start_time
    print(f"run time: {run_time}s")
    si.pl.pbg_metrics(fig_ncol=3)
    plt.savefig(f"{workdir}/pbg_metrics.png")