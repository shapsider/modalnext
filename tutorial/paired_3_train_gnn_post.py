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

workdir = f"../dataset_repo/paired/ma2020/gnn_result_processed"

flag = "step1"

mdata = mu.read_h5mu("../dataset_repo/paired/ma2020/ma2020_preprocessed.h5mu.gz")
print(mdata)

adata_CP = mdata['atac']
adata_CG = mdata['rna']

# load in graph ('graph0') info
si.load_graph_stats(path=f'./{workdir}/pbg/graph0/')
# load in model info for ('graph0')
si.load_pbg_config(path=f'./{workdir}/pbg/graph0/model/')

dict_adata = si.read_embedding()
adata_C = dict_adata['C']  # embeddings for cells
adata_G = dict_adata['G']  # embeddings for genes
adata_P = dict_adata['P']  # embeddings for peaks

if flag == "step1":
    adata_C.obs['celltype'] = adata_CG[adata_C.obs_names,:].obs['celltype'].copy()
    si.tl.umap(adata_C,n_neighbors=15,n_components=2)
    adata_C.obsm["X_emb"] = adata_C.X
    print("adata_C: ", adata_C)
    # adata_C.uns['celltype_colors'] = ['#F8D856', '#F1B044', '#C37777', '#897a74', "#d6a780"]
    adata_C.write(f"{workdir}/adata_C.h5ad")

    # genes
    adata_cmp_CG = si.tl.compare_entities(adata_ref=adata_C,
                                        adata_query=adata_G)
    # peaks
    adata_cmp_CP = si.tl.compare_entities(adata_ref=adata_C,
                                        adata_query=adata_P)

    adata_cmp_CP.var.index = \
    pd.Series(adata_cmp_CP.var.index).replace(
        to_replace=['chr3_131018470_131018770', 'chr3_131104928_131105228', 'chr3_131177880_131178180', 'chr3_131212270_131212570',
                    'chr15_102832980_102833280', 'chr15_102855927_102856227'],
        value=['Peak1(Lef1)', 'Peak2(Lef1)', 'Peak3(Lef1)', 'Peak4(Lef1)',
            'Peak1(Hoxc13)', 'Peak2(Hoxc13)'])
    
    genes_selected = adata_cmp_CG.var[(adata_cmp_CG.var['max']>1.5) & (adata_cmp_CG.var['gini']>0.35)].index.tolist()
    
    adata_all = sc.read(f"{workdir}/adata_all.h5ad")
    
    marker_genes = ['Top2a','Shh','Krt27','Foxq1', 'Krt31','Krt71', 'Lef1', 'Hoxc13']
    query_result = si.tl.query(adata_all,
                           obsm=None,
                           entity=marker_genes,
                           k=1000,use_radius=False,
                           anno_filter='entity_anno',
                           filters=['peak'])
    query_result.head()
    peaks_selected = list(query_result.index.unique())
    adata_all_selected = adata_all[adata_C.obs_names.to_list()
                               + genes_selected+peaks_selected].copy()
    si.tl.umap(adata_all_selected,n_neighbors=50,n_components=2)
    adata_all_selected.write(f'{workdir}/adata_all_selected.h5ad')