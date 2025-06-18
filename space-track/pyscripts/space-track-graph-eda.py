import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_nodes_dataframe(dataset_path, reduced=False, frac1=0.25, reduced_sample_alt_e=True, frac2=1.0, min_alt=500,
                        max_alt=520,
                        e_thres=0.25, sampled1=False, reduced_sample_leos=False, frac3=0.25, leo='leo4', sampled2=True):
    # Determine the file path based on conditions
    if reduced:
        nodes_savepath = f"{dataset_path}datasets/space-track-ap2-graph-node-feats-reduced-{int(frac1 * 100)}.csv"
    elif reduced_sample_alt_e:
        if sampled1:
            nodes_savepath = f"{dataset_path}datasets/space-track-ap2-graph-node-feats-reduced-{int(frac2 * 100)}-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
        else:
            nodes_savepath = f"{dataset_path}datasets/space-track-ap2-graph-node-feats-reduced-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
    elif reduced_sample_leos:
        if sampled2:
            nodes_savepath = f"{dataset_path}datasets/space-track-ap2-graph-node-feats-{leo}-reduced-{int(frac3 * 100)}.csv"
        else:
            nodes_savepath = f"{dataset_path}datasets/space-track-ap2-graph-node-feats-{leo}.csv"
    else:
        nodes_savepath = f'{dataset_path}datasets/space-track-ap2-graph-node-feats.csv'

    # Read the CSV file and clean the data
    nodes_df = pd.read_csv(nodes_savepath, memory_map=True,
                           usecols=['NORAD_CAT_ID', 'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE',
                                    'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'REV_AT_EPOCH', 'BSTAR', 'MEAN_MOTION_DOT',
                                    'SEMIMAJOR_AXIS', 'PERIOD', 'APOAPSIS', 'PERIAPSIS',
                                    'OBJECT_TYPE', 'RCS_SIZE', 'CONSTELLATION_DISCOS_ID', 'PX', 'PY', 'PZ',
                                    'VX', 'VY', 'VZ', 'TIMESTAMP']).set_index('NORAD_CAT_ID').fillna(
        {'CONSTELLATION_DISCOS_ID': 0})

    print(f"Space-Track AP2 graph node features loaded from: {nodes_savepath}")

    nodes_df[['EPOCH_DATE', 'EPOCH_TIME']] = nodes_df['TIMESTAMP'].str.split(' ', expand=True)
    nodes_df['CONSTELLATION_DISCOS_ID'] = nodes_df['CONSTELLATION_DISCOS_ID'].astype(int)
    # Return the dataframe
    return nodes_df, nodes_savepath


def print_graph_statistics(edges_df):
    datetime_grouped_edges_df = edges_df.groupby('timestamp')
    summary = datetime_grouped_edges_df.agg(
        num_edges=('source', 'size'),
        num_unique_src=('source', 'nunique'),
        num_unique_tgt=('target', 'nunique')
    )

    summary['num_nodes'] = datetime_grouped_edges_df.apply(
        lambda x: len(set(x['source']) | set(x['target'])),
        include_groups=False
    )

    summary['graph_density'] = datetime_grouped_edges_df.apply(
        lambda x: len(x) / (len(set(x['source']) | set(x['target'])) * (len(set(x['source']) | set(x['target'])) - 1))
        if len(set(x['source']) | set(x['target'])) > 1 else 0,
        include_groups=False
    )

    # Compute statistics
    stats = summary[['num_edges', 'num_nodes', 'graph_density']].agg(['max', 'min', 'mean', 'std'])

    # Print statistics with phrases
    print('Graph Statistics:')
    print('#' * 50)
    print(f"Maximum number of edges: {stats.loc['max', 'num_edges']}")
    print(f"Minimum number of edges: {stats.loc['min', 'num_edges']}")
    print(f"Mean number of edges: {stats.loc['mean', 'num_edges']:.2f}")
    print(f"Standard deviation of edges: {stats.loc['std', 'num_edges']:.2f}")
    print('#' * 50)
    print(f"Maximum number of nodes: {stats.loc['max', 'num_nodes']}")
    print(f"Minimum number of nodes: {stats.loc['min', 'num_nodes']}")
    print(f"Mean number of nodes: {stats.loc['mean', 'num_nodes']:.2f}")
    print(f"Standard deviation of nodes: {stats.loc['std', 'num_nodes']:.2f}")
    print('#' * 50)
    print(f"Maximum graph density: {stats.loc['max', 'graph_density']:.5f}")
    print(f"Minimum graph density: {stats.loc['min', 'graph_density']:.5f}")
    print(f"Mean graph density: {stats.loc['mean', 'graph_density']:.5f}")
    print(f"Standard deviation of graph density: {stats.loc['std', 'graph_density']:.5f}")

    # --- Compute node degree statistics ---
    # For each timestamp group, compute the degree for each node as the sum of counts in 'source' and 'target'
    def node_degree_stats(group):
        # Compute counts from source and target separately and combine
        src_counts = group['source'].value_counts()
        tgt_counts = group['target'].value_counts()
        # Combine the two count series: nodes that appear in either source or target
        all_nodes = set(group['source']).union(set(group['target']))
        degrees = {node: src_counts.get(node, 0) + tgt_counts.get(node, 0) for node in all_nodes}
        degree_values = list(degrees.values())
        return pd.Series({
            'max_degree': max(degree_values),
            'min_degree': min(degree_values),
            'mean_degree': np.mean(degree_values),
            'std_degree': np.std(degree_values, ddof=1) if len(degree_values) > 1 else 0
        })

    # Apply the node degree function on each timestamp group
    degree_summary = datetime_grouped_edges_df.apply(node_degree_stats, include_groups=False)

    # Aggregate the node degree stats across timestamps
    degree_stats = degree_summary.agg(['max', 'min', 'mean', 'std'])

    # Print node degree statistics
    print('#' * 50)
    print(f"Maximum node degree (across groups' max_degree): {degree_stats.loc['max', 'max_degree']}")
    print(f"Minimum node degree (across groups' min_degree): {degree_stats.loc['min', 'min_degree']}")
    print(f"Mean of average node degree (mean of mean_degree): {degree_stats.loc['mean', 'mean_degree']:.2f}")
    print(
        f"Standard deviation of average node degree (std of mean_degree): {degree_stats.loc['std', 'mean_degree']:.2f}")


def compute_average_active_time(edges_df):
    """
    Computes the average active time of all edges in the graph.
    An active time of an edge corresponds to the maximum number of consecutive timestamps
    the edge appears throughout the dataset.

    :param edges_df: A pandas DataFrame with columns ['source', 'target', 'timestamp'].
    :return: The average active time of all edges.
    """
    # Convert timestamp column to datetime
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])

    # Sort values by edge and timestamp
    edges_df = edges_df.sort_values(by=['source', 'target', 'timestamp'])

    # Group by edges
    edge_groups = edges_df.groupby(['source', 'target'])['timestamp'].apply(list)

    active_times = []

    for timestamps in edge_groups:
        timestamps = sorted(timestamps)  # Ensure timestamps are sorted
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(timestamps)):
            # Compute time difference in hours (assuming hourly intervals)
            time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600

            if time_diff == 1:  # Consecutive timestamp
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1  # Reset count

        # Update max consecutive appearances
        max_consecutive = max(max_consecutive, current_consecutive)
        active_times.append(max_consecutive)

    # Compute average active time
    return sum(active_times) / float(len(active_times)) if active_times else 0


def plot_edge_counts_by_windows(edges_df, window_sizes=['1min', '5min', '15min'], save=False, savepath=None):
    """e
    - edges_df: pandas DataFrame with columns ['source', 'target', 'timestamp'].
      The 'timestamp' column should be convertible to datetime.
    - window_sizes: list of strings representing pandas offset aliases (e.g., '1min', '5min', etc.).

    The function creates a line plot for each window size, making it easier to decide
    which window size yields a reasonable balance between noise and smoothing for your
    time-series model.
    """
    # Ensure the timestamp column is in datetime format
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])

    # List to store the aggregated dataframes for each window size
    agg_dfs = []
    for window in window_sizes:
        # Resample based on the window size and count the number of edges in each window
        df_window = (edges_df
                     .set_index('timestamp')
                     .resample(window)
                     .size()
                     .reset_index(name='edge_count'))
        # Add a column to identify the window size used
        df_window['window_size'] = window
        agg_dfs.append(df_window)

    # Combine all the aggregated data into one dataframe for plotting
    agg_df = pd.concat(agg_dfs)

    # Create the line plot using seaborn
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=agg_df, x='timestamp', y='edge_count', hue='window_size', marker="o")
    # plt.title('Edge Counts Aggregated by Different Window Sizes')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Edges')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(f'{savepath}edge_counts_by_windows.pdf', format='pdf')

    plt.show()


def plot_graph_dynamics(edges_df, nodes_df, save=False, savepath=None):
    # Ensure the timestamp columns are in datetime format.
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])
    nodes_df['TIMESTAMP'] = pd.to_datetime(nodes_df['TIMESTAMP'])

    # Sort the edges by time
    edges_df = edges_df.sort_values('timestamp')

    # Determine the overall time range
    start_time = edges_df['timestamp'].min()
    end_time = edges_df['timestamp'].max()

    # Define window sizes in minutes from 5 to 300 (inclusive) with 5 minute steps
    window_sizes = np.arange(5, 305, 5)
    metric_values = []

    # Loop over each window size
    for w in window_sizes:
        window_metric_list = []
        current_start = start_time

        # Divide the overall time range into non-overlapping windows of size w minutes.
        while current_start < end_time:
            current_end = current_start + pd.Timedelta(minutes=w)
            # Filter the edges for this window
            window_edges = edges_df[(edges_df['timestamp'] >= current_start) & (edges_df['timestamp'] < current_end)]
            if window_edges.empty:
                current_start = current_end
                continue

            # Get the sorted unique timestamps in this window
            times_in_window = sorted(window_edges['timestamp'].unique())

            # If there's not enough data (at least 2 timestamps), skip this window.
            if len(times_in_window) < 2:
                current_start = current_end
                continue

            pair_metrics = []
            # Compute the metric for each consecutive pair of timestamps
            for i in range(len(times_in_window) - 1):
                t1, t2 = times_in_window[i], times_in_window[i + 1]
                # Get the sets of edges at these two timestamps.
                edges_t1 = set(window_edges[window_edges['timestamp'] == t1][['source', 'target']].apply(tuple, axis=1))
                edges_t2 = set(window_edges[window_edges['timestamp'] == t2][['source', 'target']].apply(tuple, axis=1))

                # Compute the symmetric difference and union.
                union_edges = edges_t1.union(edges_t2)
                if len(union_edges) == 0:
                    diff_ratio = 0
                else:
                    diff_edges = edges_t1.symmetric_difference(edges_t2)
                    diff_ratio = len(diff_edges) / len(union_edges)
                pair_metrics.append(diff_ratio)

            # Average the metric for this window if we computed any pair metrics.
            if pair_metrics:
                window_metric_list.append(np.mean(pair_metrics))

            # Move to the next window.
            current_start = current_end

        # Average the metric over all windows for this window size.
        if window_metric_list:
            metric_values.append(np.mean(window_metric_list))
        else:
            metric_values.append(np.nan)

    # Create the plot using seaborn.
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x=window_sizes, y=metric_values, marker="o")
    ax.set_xlabel("Window Size (minutes)")
    ax.set_ylabel("Average Edge Dynamics Change Metric")
    # ax.set_title("Graph Dynamics Metric vs. Window Size")
    plt.tight_layout()
    if save:
        plt.savefig(f'{savepath}graph_dynamics_by_window_size.pdf', format='pdf')
    plt.show()


def plot_graph_stability(edges_df, nodes_df, save=False, savepath=None):
    # Convert timestamp columns to datetime objects.
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])
    nodes_df['TIMESTAMP'] = pd.to_datetime(nodes_df['TIMESTAMP'])

    # Sort the edges by time.
    edges_df = edges_df.sort_values('timestamp')

    # Determine the overall time range.
    start_time = edges_df['timestamp'].min()
    end_time = edges_df['timestamp'].max()

    # Define window sizes in minutes from 5 to 300 (inclusive) with 5 minute steps.
    window_sizes = np.arange(5, 305, 5)
    metric_values = []

    # Loop over each window size.
    for w in window_sizes:
        window_metric_list = []
        current_start = start_time

        # Divide the overall time range into non-overlapping windows of size w minutes.
        while current_start < end_time:
            current_end = current_start + pd.Timedelta(minutes=w)
            # Filter the edges for this window.
            window_edges = edges_df[(edges_df['timestamp'] >= current_start) & (edges_df['timestamp'] < current_end)]
            if window_edges.empty:
                current_start = current_end
                continue

            # Get the sorted unique timestamps in this window.
            times_in_window = sorted(window_edges['timestamp'].unique())

            # Need at least 2 timestamps to compare.
            if len(times_in_window) < 2:
                current_start = current_end
                continue

            pair_metrics = []
            # Compute the stability metric for each consecutive pair of timestamps.
            for i in range(len(times_in_window) - 1):
                t1, t2 = times_in_window[i], times_in_window[i + 1]
                # Create sets of edges for t1 and t2.
                edges_t1 = set(window_edges[window_edges['timestamp'] == t1][['source', 'target']].apply(tuple, axis=1))
                edges_t2 = set(window_edges[window_edges['timestamp'] == t2][['source', 'target']].apply(tuple, axis=1))

                union_edges = edges_t1.union(edges_t2)
                if len(union_edges) == 0:
                    stability = 0
                else:
                    intersection_edges = edges_t1.intersection(edges_t2)
                    stability = len(intersection_edges) / len(union_edges)
                pair_metrics.append(stability)

            # Average the metric for this window if we computed any pair metrics.
            if pair_metrics:
                window_metric_list.append(np.mean(pair_metrics))

            # Move to the next window.
            current_start = current_end

        # Average the metric over all windows for this window size.
        if window_metric_list:
            metric_values.append(np.mean(window_metric_list))
        else:
            metric_values.append(np.nan)

    # Plot the metric using seaborn.
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x=window_sizes, y=metric_values, marker="o")
    ax.set_xlabel("Window Size (minutes)")
    ax.set_ylabel("Average Edge Stability Metric")
    # ax.set_title("Graph Edge Stability vs. Window Size")
    plt.tight_layout()
    if save:
        plt.savefig(f'{savepath}graph_stability_by_window_size.pdf', format='pdf')
    plt.show()


def main():
    dataset_path = '/data/f.caldas/ssk/'
    savepath = 'figures/satcon/'
    save = True
    num_satellites = 5

    print("Loading data...", flush=True)
    load_start = time.time()
    nodes_df, nodes_savepath = get_nodes_dataframe(dataset_path)
    edges_df = pd.read_csv(nodes_savepath.replace('node-feats', 'edges'), memory_map=True).rename(
        columns={'datetime': 'timestamp'})
    load_end = time.time()

    print("$" * 101)
    print(f"Number of satellites in the graph: {nodes_df.index.unique().shape[0]}", flush=True)
    print_graph_statistics(edges_df)
    print(f'Number of time points in the graph: {nodes_df["TIMESTAMP"].unique().shape[0]}', flush=True)
    print(f"Number of constellations in the graph: {nodes_df['CONSTELLATION_DISCOS_ID'].unique().shape[0]}",
          flush=True)

    orbital_units = {
        "MEAN_MOTION": "rev/day",
        "ECCENTRICITY": "",
        "INCLINATION": "degrees",
        "RA_OF_ASC_NODE": "degrees",
        "ARG_OF_PERICENTER": "degrees",
        "MEAN_ANOMALY": "degrees",
        "BSTAR": "earth radii^-1",
        "MEAN_MOTION_DOT": "rev/day^2",
        "MEAN_MOTION_DDOT": "rev/day^3",
        "SEMIMAJOR_AXIS": "km",
        "PERIOD": "minutes",
        "APOAPSIS": "km",
        "PERIAPSIS": "km"
    }
    # numerical_columns = nodes_df.select_dtypes(include='float').columns
    # categorical_columns = nodes_df.select_dtypes(include=['object', 'int']).columns.drop(
    #     ['REV_AT_EPOCH', 'TIMESTAMP', 'EPOCH_DATE', 'EPOCH_TIME'])
    # log_scale_columns = ['ECCENTRICITY', 'BSTAR', 'MEAN_MOTION_DOT']

    sns.set(style="whitegrid")

    avg_active_time = compute_average_active_time(edges_df)
    print(f"Average active time of edges: {avg_active_time:.2f} hours")

    plot_edge_counts_by_windows(edges_df, window_sizes=['5min', '30min', '60min', '128min', '256min'], save=save,
                                savepath=savepath)

    plot_graph_dynamics(edges_df, nodes_df, save=save, savepath=savepath)

    plot_graph_stability(edges_df, nodes_df, save=save, savepath=savepath)

    print("Plotting...", flush=True)
    print("$" * 101)
    print("Plotting distributions of categorical columns...", flush=True)
    plot_start = time.time()

    plot_end = time.time()

    print("$" * 101)
    load_time = (load_end - load_start) / 60
    plot_time = (plot_end - plot_start) / 60
    print(f"Time taken to load the data: {load_time:.2f} minutes", flush=True)
    print(f"Time taken to plot the data: {plot_time:.2f} minutes", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
