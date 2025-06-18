# MSc Thesis
Official repository for MSc Thesis titled: `Causal Inference Explanations for Graph Neural Networks`.

## Conda Installation
To set up the environment, follow these steps:

1. **Using `env.yml` (recommended):**

    ```bash
    conda env create -f env.yml
    conda activate ciexplainer
    ```

2. **If the above does not work, use `requirements.txt`:**

    ```bash
    conda create -n ciexplainer python=3.9
    conda activate ciexplainer
    pip install -r requirements.txt
    ```

## Running the explainers

The python `gnn_explain_std.py` file, is used for explaining Graph Neural Network (GNN) predictions for all datasets, all GNNs of different tasks and using different explainers.

Usage:
```bash
    python gnn_explain_std.py --task $TASK --model all --dataset all --explainer $EXPLAINER --num_runs $RUNS --std_idx $STD
```
Arguments:
    
    --task      Task to explain: node classification (nc), graph classification (gc), link prediction (lp), or all.
    
    --explainer Explainer to use: random, gnnexplainer, pgexplainer, subgraphX, ciexplainer, or all.

    --num_runs  Number of experiments runs.

    --std_idx Index of the standard deviation value to use.
    

Example:

```bash
    python gnn_explain_std.py --task nc --model all --dataset all --explainer ciexplainer --num_runs 5 --std_idx 4
```
