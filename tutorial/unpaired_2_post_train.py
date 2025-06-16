import os
import sys
sys.path.append("..")
import src.graph_src as si
import anndata
import matplotlib.pyplot as plt
import time 
import pandas as pd
import muon as mu
import scanpy as sc
import scib
import math
import warnings
warnings.filterwarnings("ignore")

dataset = "pbmc"

workdir = f"../dataset_repo/unpaired/{dataset}/graph"
result_path = f"../dataset_repo/unpaired/{dataset}"

# load in graph ('graph0') info
si.load_graph_stats(path=f'./{workdir}/pbg/graph_full/')
# load in model info for ('graph0')
si.load_pbg_config(path=f'./{workdir}/pbg/graph_full/init_model/')

dict_adata = si.read_embedding()
adata_C = dict_adata['C']  # embeddings for cells
adata_G = dict_adata['G']  # embeddings for genes
adata_P = dict_adata['P']  # embeddings for peaks

adata_C = dict_adata['C']
adata_C2 = dict_adata['C2']

adata_CG = anndata.read_h5ad(f"../dataset_repo/unpaired/{dataset}/rna_hvg.h5ad")
adata_CP = anndata.read_h5ad(f"../dataset_repo/unpaired/{dataset}/atac_hvg.h5ad")

adata_CP.obs.index = adata_CP.obs.index + '_atac'
adata_CG.obs.index = adata_CG.obs.index + '_rna'

adata_C.obs['cell_type'] = adata_CP[adata_C.obs_names,:].obs['cell_type']
adata_C2.obs['cell_type'] = adata_CG[adata_C2.obs_names,:].obs['cell_type']

combined = adata_C2.concatenate(adata_C, batch_categories=['RNA', 'ATAC'])
sc.tl.pca(combined, n_comps=10, svd_solver="auto")
sc.pp.neighbors(combined, 
                # n_pcs=10, 
                use_rep="X_pca", metric="cosine")
sc.tl.umap(combined)

print(combined)

combined.write(f"{result_path}/combined.h5ad", compression="gzip")