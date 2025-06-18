import argparse
import pandas as pd
from tabulate import tabulate

from explainer_store import explainer_names

def compute_mean_std(csv_files, output_file):
    """
    Reads multiple CSV files, computes the mean and standard deviation of each column,
    and prints the results in a formatted table.

    Parameters:
        csv_files (list): List of file paths to CSV files.
    """
    # Read all dataframes into a list
    dataframes = [pd.read_csv(file, header=[0, 1], index_col=[0, 1]) for file in csv_files]

    # Concatenate along a new axis to align indices and columns
    combined_data = pd.concat(dataframes, axis=0, keys=range(len(csv_files)), names=['Run'])

    # Compute mean and std
    mean_df = combined_data.groupby(level=[1, 2]).mean()
    std_df = combined_data.groupby(level=[1, 2]).std()

    # Create formatted results
    formatted_results = mean_df.copy()
    for col in mean_df.columns:
        formatted_results[col] = mean_df[col].map(lambda x: f"{x:.4f}") + " ± " + std_df[col].map(lambda x: f"{x:.4f}")

    # Convert to tabulated format
    table_str = tabulate(formatted_results, headers='keys', tablefmt='grid')
    # Save to text file
    with open(output_file, "w") as f:
        f.write("Results Summary (Mean ± Std)\n\n")
        f.write(table_str)

    print(f"Results saved to {output_file}")

    return mean_df, std_df, combined_data


def main(task, nruns=10):
    eval_metrics_path = 'eval_metrics/'
    results_path = 'results/'
    runs = nruns

    if task in ['nc', 'all']:
        print('Aggregating results for node classification task...')
        task_filename = '_nc_metrics_'
        for explainer_name in explainer_names:
            if explainer_name not in ['subgraphx', 'cf_gnnexplainer']:
                csv_files = [f'{eval_metrics_path}{explainer_name}{task_filename}{run}.csv' for run in range(runs)]
                output_file = f'{results_path}{explainer_name}{task_filename}agg.txt'
                compute_mean_std(csv_files, output_file)

    if task in ['gc', 'all']:
        print('Aggregating results for graph classification task...')
        task_filename = '_gc_metrics_'
        for explainer_name in explainer_names:
            if explainer_name not in ['cf_gnnexplainer']:
                csv_files = [f'{eval_metrics_path}{explainer_name}{task_filename}{run}.csv' for run in range(runs)]
                output_file = f'{results_path}{explainer_name}{task_filename}agg.txt'
                compute_mean_std(csv_files, output_file)
    if task in ['lp']:
        print('Aggregating results for link prediction task...')
        task_filename = '_lp_metrics_'
        for explainer_name in explainer_names:
            csv_files = [f'{eval_metrics_path}{explainer_name}{task_filename}{run}.csv' for run in range(runs)]
            output_file = f'{results_path}{explainer_name}{task_filename}agg.txt'
            compute_mean_std(csv_files, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate results from multiple runs.')
    parser.add_argument('--task', type=str, default='all', help='Task to aggregate results for. Options: nc, gc, lp, all')
    parser.add_argument('--nruns', type=int, default=10, help='Number of runs to aggregate.')
    args = parser.parse_args()
    main(args.task, args.nruns)
