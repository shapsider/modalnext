import os
import sys
sys.path.append("..")
import src.graph_src as si
import anndata
import matplotlib.pyplot as plt
import time 

dataset = "pbmc"

workdir = f"../dataset_repo/unpaired/{dataset}/graph"
si.settings.set_workdir(workdir)
result_path = f"../dataset_repo/unpaired/{dataset}"

adata_CG = anndata.read_h5ad(f"../dataset_repo/unpaired/{dataset}/rna_hvg.h5ad")
adata_CP = anndata.read_h5ad(f"../dataset_repo/unpaired/{dataset}/atac_hvg.h5ad")
print(adata_CG)
print(adata_CP)

adata_CG.X = adata_CG.layers["counts"]
adata_CP.X = adata_CP.layers["counts"]

rna_counts = adata_CG.layers["counts"].copy()
atac_counts = adata_CP.layers["counts"].copy()

adata_CP.obs.index = adata_CP.obs.index + '_atac'
adata_CG.obs.index = adata_CG.obs.index + '_rna'

adata_CP.var['chr'] = adata_CP.var['chrom']
adata_CP.var['start'] = adata_CP.var['chromStart']
adata_CP.var['end'] = adata_CP.var['chromEnd']

si.pp.normalize(adata_CG,method='lib_size')
si.pp.log_transform(adata_CG)
si.tl.discretize(adata_CG,n_bins=5)
si.pl.discretize(adata_CG,kde=True)
plt.savefig(f"{workdir}/discretize_rna.png")

# mm10
# hg38
print(adata_CP)
adata_CG_atac = si.tl.gene_scores(adata_CP,genome='hg38',use_gene_weigt=True, use_top_pcs=False)

si.pp.filter_genes(adata_CG_atac,min_n_cells=3)
si.pp.cal_qc_rna(adata_CG_atac)
si.pp.normalize(adata_CG_atac,method='lib_size')
si.pp.log_transform(adata_CG_atac)

adata_CrnaCatac = si.tl.infer_edges(adata_CG, adata_CG_atac, n_components=15, k=15)

si.pl.node_similarity(adata_CrnaCatac,cutoff=0.5)
plt.savefig(f"{workdir}/node_similarity.png")

si.pl.svd_nodes(adata_CrnaCatac,
                color=['cell_type'],
                size=3,
                cutoff=0.5,
                fig_legend_ncol=2)
plt.savefig(f"{workdir}/svd_nodes.png")

si.tl.trim_edges(adata_CrnaCatac, cutoff=0.5)

# ATAC预处理
si.pp.cal_qc_rna(adata_CP)
si.pp.normalize(adata_CP,method='lib_size')
si.pp.log_transform(adata_CP)
si.tl.discretize(adata_CP,n_bins=5)
si.pl.discretize(adata_CP,kde=False)
plt.savefig(f"{workdir}/discretize_atac.png")

si.tl.gen_graph(list_CP=[adata_CP],
                list_CG=[adata_CG],
                list_CC=[adata_CrnaCatac],
                copy=False,
                use_highly_variable=False,
                use_top_pcs=False,
                dirname='graph_full')

print(si.settings.pbg_params)

if __name__ == '__main__':
    starttime = time.time()
    si.tl.pbg_train(auto_wd=True, save_wd=True, output='init_model')
    endtime = time.time()
    print('spent {:.5f} s'.format(endtime-starttime))

    si.pl.pbg_metrics(fig_ncol=3)
    plt.savefig(f"{workdir}/pbg_metrics.png")

    dict_adata = si.read_embedding()
    adata_C = dict_adata['C']
    adata_C2 = dict_adata['C2']
    adata_all = si.tl.embed(adata_ref=adata_C2,
                        list_adata_query=[adata_C],
                        use_precomputed=False)
    adata_CG.obsm['X_emb'] = adata_all[adata_CG.obs_names,:].X
    adata_CP.obsm['X_emb'] = adata_all[adata_CP.obs_names,:].X

    adata_CG.layers['counts'] = rna_counts
    adata_CP.layers['counts'] = atac_counts

    print(adata_CG.layers['counts'].max())
    print(adata_CP.layers['counts'].max())

    adata_CG.write(f"{result_path}/rna_result.h5ad", compression="gzip")
    adata_CP.write(f"{result_path}/atac_result.h5ad", compression="gzip")