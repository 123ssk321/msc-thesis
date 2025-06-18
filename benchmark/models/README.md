# Trained Graph Neural Networks

This directory contains the GNNs used in the thesis divided into subdirectories named after the dataset where they were trained.

**Synthetic Datasets**: There 10 versions of each GNN for each dataset, with different standard deviations for the artificially sampled node features. Naming convention: `[task]_[gnn]_[dataset][X]_model.pth`, where `X` is the standard deviation and can have one of the following values: '`00`', '`01`', '`02`', '`03`', '`04`', '`05`', '`06`', '`07`', '`08`', '`09`', or '`10`'. Note that each value of `X` corresponds to a floating point standard deviation, respectively, `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`, `0.8`, `0.9`, or `1.0`.

`task` can take one of the following values: '`nc`', '`gc`', '`lp`', corresponding respectively to, node classification, graph classification and link prediction.

`gnn` can take one of the following values: '`gcn`', '`graphsage`', '`gat`', '`gin`', corresponding respectively to, Graph Convolution Network, GraphSAGE, Graph Attention Network and Graph Isomorphism Network.

`dataset` can take one of the following values: '`ba_shapes`', '`tree_grid`', '`ba_2motif`'.

**Real Datasets**: MUTAG (graph classification) and Cora (link prediction). There is no variation of GNNs for these datasets. The naming convention it is the same as in the synthetic case but without `X` and `dataset` must have the value '`mutag`' or '`cora`'.

