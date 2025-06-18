# Datasets

This directory contains the publicly available benchmark datasets used in the thesis.

**Synthetic Datasets**: BA-Shapes (node classification), Tree-Grid(node classification), BA-2motif (graph classification), BA-Shapes-Link (link prediction), Tree-Grid-Link (link prediction). There 10 versions of each dataset, with different standard deviations for the artificially sampled node features. Naming convention: `[dataset][X].pth`, where `X` is the standard deviation and can have one of the following values: '`00`', '`01`', '`02`', '`03`', '`04`', '`05`', '`06`', '`07`', '`08`', '`09`', or '`10`'. Note that each value of `X` corresponds to a floating point standard deviation, respectively, `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`, `0.8`, `0.9`, or `1.0`.

`dataset` can take one of the following values: '`ba_shapes`', '`tree_grid`', '`ba_2motif`'.

**Real Datasets**: MUTAG (graph classification) and Cora (link prediction).  There are no variations of these datasets because they have original features.