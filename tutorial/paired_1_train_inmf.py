import sys
sys.path.append("..") 
import src.nmf as nmf
import muon as mu
import scanpy as sc
import os
import matplotlib.pyplot as plt
import time

figure_directory = "../dataset_repo/paired/ma2020/inmf"
if not os.path.exists(figure_directory):
    os.makedirs(figure_directory)

mdata = mu.read_h5mu("../dataset_repo/paired/ma2020/ma2020_preprocessed.h5mu.gz")

sc.pp.subsample(mdata["rna"], n_obs=1000, random_state=0)
sc.pp.subsample(mdata["atac"], n_obs=1000, random_state=0)
# Create a MuData object with rna and atac.
mdata = mu.MuData({"rna": mdata["rna"], "atac": mdata["atac"]})
print(mdata)
print(type(mdata['rna'].X))
print(mdata['rna'].X.max())
print(mdata['rna'].X.min())
print(type(mdata['atac'].X))
print(mdata['atac'].X.max())
print(mdata['atac'].X.min())

# Umap RNA
sc.pp.scale(mdata["rna"], zero_center=False)
sc.tl.pca(mdata["rna"], svd_solver="arpack")
sc.pp.neighbors(mdata["rna"], n_neighbors=10, n_pcs=10)
sc.tl.umap(mdata["rna"], spread=1.5, min_dist=0.5)

# Umap ATAC
sc.pp.scale(mdata["atac"], zero_center=False)
sc.tl.pca(mdata["atac"], svd_solver="arpack")
sc.pp.neighbors(mdata["atac"], n_neighbors=10, n_pcs=10)
sc.tl.umap(mdata["atac"], spread=1.5, min_dist=0.5)

mdata.write(f"{figure_directory}/ma2020_step1.h5mu.gz")

# Define the model.
model = nmf.models.nmfModel(
    latent_dim=20,
    h_regularization=1e-2,
    w_regularization=1e-3,
    eps=0.1,
    cost="cosine",
)

start_time = time.time()
model.train(mdata, 
            device="cuda:0"
            )
end_time = time.time()
run_time = end_time - start_time
print(f"run time: {run_time}s")

sc.pp.neighbors(mdata, use_rep="W_OT", key_added="nmf")
sc.tl.umap(mdata, neighbors_key="nmf")
sc.pl.umap(mdata["rna"], color="celltype")
print(mdata)
plt.savefig(f"{figure_directory}/integration.png")

mdata.write(f"{figure_directory}/nmf.h5mu.gz")