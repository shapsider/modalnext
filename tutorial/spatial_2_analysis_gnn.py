import os
import sys
sys.path.append("..")
import src.graph_src as si
import anndata
import matplotlib.pyplot as plt
import time 
import scanpy as sc
import pandas as pd

from src.spatialknn.utils import clustering

k_number = 3
workdir = f"../dataset_repo/spatial/Mouse_Thymus/gnn_result_{k_number}nn"

adata_CG = sc.read("../dataset_repo/spatial/Mouse_Thymus/adata_RNA.h5ad")
adata_CPro = sc.read("../dataset_repo/spatial/Mouse_Thymus/adata_ADT.h5ad")

# load in graph ('graph0') info
si.load_graph_stats(path=f'./{workdir}/pbg/graph_spatial/')
# load in model info for ('graph0')
si.load_pbg_config(path=f'./{workdir}/pbg/graph_spatial/model/')

dict_adata = si.read_embedding()
print(dict_adata)
adata_C = dict_adata['C']  # embeddings for cells
adata_G = dict_adata['G']  # embeddings for genes
adata_Pro = dict_adata['Pro']  # embeddings for peaks

print(adata_C)
print(adata_C.obs_names)
print(adata_CG.obs_names)

adata_C.obsm["emb"] = adata_C.X
adata_C.obsm['spatial'] = adata_CG[adata_C.obs_names,:].obsm['spatial'].copy()

tool = 'mclust' # mclust, leiden, and louvain
clustering(adata_C, key='emb', add_key='clusters_simba', n_clusters=8, method=tool, use_pca=False)

fig, ax_list = plt.subplots(1, 2, figsize=(14, 5))
sc.pp.neighbors(adata_C, use_rep='emb', n_neighbors=30)
sc.tl.umap(adata_C)
sc.pl.umap(adata_C, color='clusters_simba', ax=ax_list[0], s=60, show=False)

adata_C.obsm['spatial'][:,1] = -1*adata_C.obsm['spatial'][:,1]
sc.pl.embedding(adata_C, basis='spatial', color='clusters_simba', ax=ax_list[1], s=90, show=False)

plt.tight_layout(w_pad=0.3)
plt.savefig(f"{workdir}/cells_{tool}.png")

# annotation
adata_C.obs['clusters_simba_number'] = adata_C.obs['clusters_simba'].copy()
adata_C.obs['clusters_simba'].cat.rename_categories({4: '5-Outer cortex region 3(DN T,DP T,cTEC)',
                                                2: '7-Subcapsular zone(DN T)',
                                                6: '4-Middle cortex region 2(DN T,DP T,cTEC)',
                                                8: '2-Corticomedullary Junction(CMJ)',
                                                5: '1-Medulla(SP T,mTEC,DC)',
                                                1: '6-Connective tissue capsule(fibroblast)',
                                                3: '8-Connective tissue capsule(fibroblast,RBC,myeloid)',
                                                7: '3-Inner cortex region 1(DN T,DP T,cTEC)'
                                                }, inplace=True)
list_ = ['3-Inner cortex region 1(DN T,DP T,cTEC)','2-Corticomedullary Junction(CMJ)','4-Middle cortex region 2(DN T,DP T,cTEC)',
         '7-Subcapsular zone(DN T)', '5-Outer cortex region 3(DN T,DP T,cTEC)', '8-Connective tissue capsule(fibroblast,RBC,myeloid)',
         '1-Medulla(SP T,mTEC,DC)','6-Connective tissue capsule(fibroblast)']
adata_C.obs['clusters_simba']  = pd.Categorical(adata_C.obs['clusters_simba'],
                      categories=list_,
                      ordered=True)
# plotting with annotation
fig, ax_list = plt.subplots(1, 2, figsize=(20, 5))
sc.pl.umap(adata_C, color='clusters_simba', ax=ax_list[0], s=60, show=False)
sc.pl.embedding(adata_C, basis='spatial', color='clusters_simba', ax=ax_list[1], s=90, show=False)

plt.tight_layout(w_pad=0.3)
plt.savefig(f"{workdir}/cells_{tool}_annotation.png")

adata_C.write(f"{workdir}/adata_C.h5ad")