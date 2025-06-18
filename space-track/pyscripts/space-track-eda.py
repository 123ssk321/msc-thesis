import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math as m
import time


def get_space_track_dataframe(dataset_path, reduced=False, frac1=0.25, reduced_sample_alt_e=True, frac2=1.0,
                              min_alt=500,
                              max_alt=520, e_thres=0.25, sampled1=False, reduced_sample_leos=False, frac3=0.25,
                              leo='leo4', sampled2=False):
    # Determine the file path based on conditions
    if reduced:
        filepath = f"{dataset_path}datasets/space-track-dataset-reduced-{int(frac1 * 100)}.csv"
    elif reduced_sample_alt_e:
        if sampled1:
            filepath = f"{dataset_path}datasets/space-track-dataset-reduced-{int(frac2 * 100)}-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
        else:
            filepath = f"{dataset_path}datasets/space-track-dataset-reduced-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
    elif reduced_sample_leos:
        if sampled2:
            filepath = f"{dataset_path}datasets/space-track-dataset-{leo}-reduced-{int(frac3 * 100)}.csv"
        else:
            filepath = f"{dataset_path}datasets/space-track-dataset-{leo}.csv"
    else:
        filepath = f'{dataset_path}datasets/space-track-dataset.csv'

    # Read the CSV file into a DataFrame
    st_df = pd.read_csv(filepath, memory_map=True,
                        usecols=['NORAD_CAT_ID', 'EPOCH_DATE', 'EPOCH_TIME', 'MEAN_MOTION', 'ECCENTRICITY',
                                 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'REV_AT_EPOCH',
                                 'BSTAR', 'MEAN_MOTION_DOT', 'SEMIMAJOR_AXIS', 'PERIOD', 'APOAPSIS', 'PERIAPSIS',
                                 'OBJECT_TYPE', 'RCS_SIZE'])
    st_df['EPOCH_TIME'] = st_df['EPOCH_TIME'].apply(lambda t: t.split('.')[0])
    st_df['TIMESTAMP'] = st_df['EPOCH_DATE'] + ' ' + st_df['EPOCH_TIME']

    print(f"Space-Track dataset loaded from: {filepath}")
    # Return the dataframe
    return st_df


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
    return nodes_df


def plot_catcol_distribution(df, column, save_path='../figures/', figsize=(8, 5), save=False, plot_title=True,
                             show=False):
    # Create a bar plot
    plt.figure(figsize=figsize)
    df_col = df[column]
    abs_values = df_col.value_counts(ascending=False)
    ax = sns.countplot(x=df_col, order=abs_values.index, stat='percent')

    rel_values = df_col.value_counts(ascending=False, normalize=True).values * 100
    lbls = [f'{p:.1f}%' for p in rel_values]

    ax.bar_label(container=ax.containers[0], labels=lbls)

    # Create the title based on the column name
    column_title = column.replace('_', ' ').title()
    title = 'Distribution of ' + column_title

    if plot_title:
        plt.title(title)
    # Set plot labels and title
    plt.xlabel(column_title)
    plt.ylabel('Percentage of Space Objects')
    plt.tight_layout()

    # Save the plot to the specified path
    if save:
        plt.savefig(f"{save_path}eda_satcon_{'_'.join(title.lower().split())}.pdf", format='pdf')

    # Display the plot
    if show:
        plt.show()
    plt.close()


def plot_catcat_distribution(df, main_column, secondary_column, save_path='../figures/', figsize=(8, 5), save=False,
                             plot_title=True, show=False):
    # Create the subset of the dataframe with the selected columns
    df_col = df[[main_column, secondary_column]]

    # Set up the figure size
    plt.figure(figsize=figsize)

    # Create the count plot
    sns.countplot(x=main_column, hue=secondary_column, data=df_col, palette='tab10')

    # Format titles
    main_column_title = main_column.replace('_', ' ').title()
    secondary_column_title = secondary_column.replace('_', ' ').title()
    title = f'{main_column_title} Distribution by {secondary_column_title}'

    # Set plot title and labels
    if plot_title:
        plt.title(title)
    plt.xlabel(main_column_title)
    plt.ylabel('Number of Space Objects')

    # Set legend
    plt.legend(title=secondary_column_title)

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PDF
    if save:
        plt.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    # Display the plot
    if show:
        plt.show()
    plt.close()


def plot_catcol_evolution_day(df, column, save_path='../figures/', figsize=(8, 6), save=False, plot_title=True,
                              show=False):
    # Prepare the column title
    column_title = column.replace('_', ' ').title()

    # Convert 'EPOCH_DATE' to datetime if it's not already in datetime format
    df['EPOCH_DATE'] = pd.to_datetime(df['EPOCH_DATE'])

    # Create a unique subset of the dataframe
    df_unique = df[['NORAD_CAT_ID', 'EPOCH_DATE', column]].drop_duplicates()
    df_unique['EPOCH_DATE'] = df_unique['EPOCH_DATE'].dt.to_period('D')

    # Group by 'EPOCH_DATE' and 'OBJECT_TYPE' to get the counts of each type per date
    df_grouped = df_unique.groupby(['EPOCH_DATE', column]).size().reset_index(name='count')

    # Convert the 'EPOCH_DATE' back to datetime for plotting
    df_grouped['EPOCH_DATE'] = df_grouped['EPOCH_DATE'].dt.to_timestamp()

    # Set up the figure size
    plt.figure(figsize=figsize)

    # Create the line plot for count evolution
    sns.lineplot(x='EPOCH_DATE', y='count', hue=column, data=df_grouped)

    # Format the title
    title = f'Evolution of {column_title} per Day'

    # Set plot title and labels
    if plot_title:
        plt.title(title)
    plt.xlabel('Epoch Date')
    plt.ylabel('Number of Space Objects')

    # Rotate the x-ticks for better readability
    plt.xticks(df_grouped['EPOCH_DATE'].unique(), rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PDF
    if save:
        plt.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    # Display the plot
    if show:
        plt.show()
    plt.close()


def plot_numcol_histogram(df, column, bins=30, unit="", log_scale=False, save_path='../figures/', figsize=(8, 5),
                          save=False, plot_title=True, show=False):
    # Set up the figure size
    plt.figure(figsize=figsize)

    # Create the histogram with KDE
    sns.histplot(df[column], log_scale=log_scale, bins=bins)
    # Format the column title
    column_title = column.replace('_', ' ').title()
    title = f'Distribution of {column_title}'

    # Set plot title and labels
    if plot_title:
        plt.title(title)

    if log_scale:
        column_title = f'log({column_title})'
    plt.xlabel(f'{column_title} ({unit})' if unit != '' else column_title)
    plt.ylabel('Number of Space Objects')
    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PDF
    if save:
        plt.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    # Display the plot
    if show:
        plt.show()
    plt.close()


def plot_numcol_violinplot(df, column, unit="", log_scale=False, nodes_df=None, save_path='../figures/', figsize=(8, 5),
                           save=False, plot_title=True, show=False):
    # Set up the figure size
    plt.figure(figsize=figsize)

    # Format the column title
    column_title = column.replace('_', ' ').title()
    title = f'Violin Plot of {column_title}'

    # Prepare data for boxplot
    if nodes_df is not None:
        # Concatenate both datasets into a single DataFrame
        data1 = df[column].values
        data2 = nodes_df[column].values

        combined_data = pd.DataFrame({
            'Values': np.concatenate([data1, data2]),
            'Dataset': ['Original'] * len(data1) + ['Propagated'] * len(data2)
        })
        sns.violinplot(hue='Dataset', y='Values', data=combined_data, log_scale=log_scale, split=True, gap=.05)
    else:
        sns.violinplot(df[column], log_scale=log_scale, split=True)

    # Set plot title and labels
    if plot_title:
        plt.title(title.replace('Violin Plot', 'Distribution'))
    if log_scale:
        column_title = f'log({column_title})'
    plt.xlabel('')
    plt.ylabel(f'{column_title} ({unit})' if unit != '' else column_title)

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PDF
    if save:
        plt.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    if show:
        plt.show()
    plt.close()


def plot_numcat_violinplot(df, numerical_column, categorical_column, nodes_df=None, unit='', log_scale=False,
                           save_path='../figures/', figsize=(10, 7), save=False, plot_title=True, show=False):
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)

    # Format the column titles
    numerical_column_title = numerical_column.replace('_', ' ').title()
    categorical_column_title = categorical_column.replace('_', ' ').title()
    title = f'Violin Plot of {numerical_column_title} by {categorical_column_title}'

    if nodes_df is not None:
        # Combine data without explicit copying: slice and assign the new column on the fly
        combined_data = pd.concat([
            df.loc[:, [categorical_column, numerical_column]].assign(Dataset='Original'),
            nodes_df.loc[:, [categorical_column, numerical_column]].assign(Dataset='Propagated')
        ], ignore_index=True)
        sns.violinplot(x=categorical_column, y=numerical_column, hue='Dataset', data=combined_data, ax=ax,
                       log_scale=log_scale, split=True, gap=.1)
    else:
        sns.violinplot(x=categorical_column, y=numerical_column, data=df, ax=ax, log_scale=log_scale, split=True)

    if plot_title:
        ax.set_title(title.replace('Violin Plot', 'Distribution'))
    ax.set_xlabel(categorical_column_title)

    if log_scale:
        numerical_column_title = f'log({numerical_column_title})'
    ax.set_ylabel(f'{numerical_column_title} ({unit})' if unit != '' else numerical_column_title)

    # Place the legend just outside the plot (to the right)
    ax.legend(title='Dataset', loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    # Adjust layout to leave space on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save:
        fig.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    if show:
        plt.show()
    plt.close()


def plot_numnumcat_scatter(df, x_column, y_column, categorical_column, x_unit='', y_unit='', x_log_scale=False,
                           y_log_scale=False,
                           save_path='../figures/', figsize=(8, 5), alpha=0.5,
                           save=False, plot_title=True, show=False):
    # Set up the figure size
    plt.figure(figsize=figsize)

    x_column_title = x_column.replace('_', ' ').title()
    y_column_title = y_column.replace('_', ' ').title()
    categorical_column_title = categorical_column.replace('_', ' ').title()

    # Construct the title using the x and y columns and the hue
    title = f'Scatter plot of {x_column_title} vs {y_column_title} by {categorical_column_title}'

    # Create the scatter plot with specified alpha transparency
    if x_log_scale:
        df[f'{x_column}_log'] = np.log(df[x_column])
        x_column = f'{x_column}_log'
        x_column_title = f'log({x_column_title})'
    if y_log_scale:
        df[f'{y_column}_log'] = np.log(df[y_column])
        y_column = f'{y_column}_log'
        y_column_title = f'log({y_column_title})'
    sns.scatterplot(x=x_column, y=y_column, hue=categorical_column, data=df, alpha=alpha)

    # Set the plot title and labels if required
    if plot_title:
        plt.title(title.replace('Scatter plot of ', ''))

    plt.xlabel(f'{x_column_title} ({x_unit})' if x_unit != '' else x_column_title)
    plt.ylabel(f'{y_column_title} ({y_unit})' if y_unit != '' else y_column_title)

    # Adjust layout for a neat appearance
    plt.tight_layout()

    # Save the plot as a PDF using a similar naming convention if requested
    if save:
        plt.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    # Display the plot
    if show:
        plt.show()
    plt.close()


def filter_valid_satellites(df, num_satellites=1):
    """
    Filters and selects valid satellites based on data availability.

    Parameters:
    df (pd.DataFrame): DataFrame containing satellite data with 'EPOCH_DATE' and 'NORAD_CAT_ID'.
    nodes_df (pd.DataFrame, optional): Additional DataFrame containing 'EPOCH_DATE'. Default is None.
    num_satellites (int, optional): Number of satellites to select. Default is 10.

    Returns:
    list: List of selected satellite NORAD_CAT_IDs.
    """
    # Ensure EPOCH_DATE is in datetime format
    df['EPOCH_DATE'] = pd.to_datetime(df['EPOCH_DATE'])

    # Identify satellites with sufficient data points
    all_dates = df['EPOCH_DATE'].unique()
    threshold = len(all_dates) // 2
    satellite_counts = df.groupby('NORAD_CAT_ID')['EPOCH_DATE'].nunique()
    valid_satellites = satellite_counts[satellite_counts > threshold].sort_values(ascending=False).index
    selected_sat_ids = valid_satellites[:num_satellites]
    return selected_sat_ids


def plot_numcol_evolution_day(df, column, sat_ids, unit='', log_scale=False, save_path='../figures/', figsize=(12, 6),
                              save=False, show=False):
    # Prepare plot title and labels
    column_title = column.replace('_', ' ').title()
    title = f'Evolution of {column_title} per Day for Space Objects'

    # Filter data for selected satellites
    plot_data = df[df['NORAD_CAT_ID'].isin(sat_ids)]

    # Create plot
    plt.figure(figsize=figsize)
    if log_scale:
        plot_data[f'{column}_log'] = np.log(plot_data[column])
        column = f'{column}_log'
        column_title = f'log({column_title})'
    sns.lineplot(x='EPOCH_DATE', y=column, hue='NORAD_CAT_ID', data=plot_data, palette='tab10')
    plt.title(title)
    plt.xlabel('Epoch Date')

    if log_scale:
        column_title = f'log({column_title})'
    plt.ylabel(f'{column_title} ({unit})' if unit != '' else column_title)
    plt.xticks(df['EPOCH_DATE'].unique(), rotation=45)

    # Place legend outside the plot area
    plt.legend(title='NORAD ID', loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure if requested
    if save:
        plt.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    if show:
        plt.show()
    plt.close()


def plot_corr_heatmap(df, columns=None, rename_dict=None, title='Correlation Matrix of Orbital Parameters',
                      save_path='../figures/',
                      figsize=(8, 5), save=False, plot_title=True, show=False):
    # Use default columns if none provided
    if columns is None:
        columns = ['INCLINATION', 'ECCENTRICITY', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER']

    # Use default renaming if none provided
    if rename_dict is None:
        rename_dict = {
            'INCLINATION': 'Inclination',
            'ECCENTRICITY': 'Eccentricity',
            'RA_OF_ASC_NODE': 'Right Ascension\nof Ascending Node',  # Split into two lines
            'ARG_OF_PERICENTER': 'Argument\nof Pericenter'  # Split into two lines
        }

    # Set up the figure size
    plt.figure(figsize=figsize)

    # Select and rename the relevant columns
    df_renamed = df[columns].rename(columns=rename_dict)

    # Generate the correlation matrix
    correlation_matrix = df_renamed.corr()

    # Create the heatmap with annotations and a coolwarm color palette
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')

    # Optionally add a title
    if plot_title:
        plt.title(title)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure as a PDF if required
    if save:
        plt.savefig(f'{save_path}eda_satcon_{"_".join(title.lower().split())}.pdf', format='pdf')

    # Display the plot
    if show:
        plt.show()
    plt.close()


def main():
    dataset_path = '/data/f.caldas/ssk/'
    savepath = 'figures/satcon/'
    save = True
    num_satellites = 5

    print("Loading data...", flush=True)
    load_start = time.time()
    st_df = get_space_track_dataframe(dataset_path)
    nodes_df = get_nodes_dataframe(dataset_path)
    load_end = time.time()

    print("$" * 101)
    print(f"Number of satellites in the dataset: {st_df['NORAD_CAT_ID'].unique().shape[0]}", flush=True)
    print(f'Number of time points in the dataset: {st_df["TIMESTAMP"].unique().shape[0]}', flush=True)
    print(f"Number of satellites in the graph: {nodes_df.index.unique().shape[0]}", flush=True)
    print(f'Number of time points in the graph: {nodes_df["TIMESTAMP"].unique().shape[0]}', flush=True)

    esa_df = pd.read_csv(f'{dataset_path}/datasets/esa-discos-satellite-data.csv',
                         usecols=['satno', 'name', 'constellationDiscosID'], memory_map=True)
    print(f"Total number of constellations: {esa_df['constellationDiscosID'].unique().shape[0]}", flush=True)

    df = pd.merge(st_df, esa_df, how='left', left_on='NORAD_CAT_ID', right_on='satno').drop('satno', axis=1).rename(
        columns={'constellationDiscosID': 'CONSTELLATION_DISCOS_ID', 'name': 'CONSTELLATION_NAME'}).fillna(
        {'CONSTELLATION_DISCOS_ID': 0})
    df['CONSTELLATION_DISCOS_ID'] = df['CONSTELLATION_DISCOS_ID'].astype(int)
    print(f"Number of constellations in the dataset: {df['CONSTELLATION_DISCOS_ID'].unique().shape[0]}", flush=True)
    print(f"Number of constellations in the graph: {nodes_df['CONSTELLATION_DISCOS_ID'].unique().shape[0]}", flush=True)

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
    numerical_columns = df.select_dtypes(include='float').columns
    categorical_columns = df.select_dtypes(include=['object', 'int']).columns.drop(
        ['NORAD_CAT_ID', 'REV_AT_EPOCH', 'TIMESTAMP', 'EPOCH_DATE', 'EPOCH_TIME', 'CONSTELLATION_NAME'])
    log_scale_columns = ['ECCENTRICITY', 'BSTAR', 'MEAN_MOTION_DOT']

    sns.set(style="whitegrid")

    print("Plotting...", flush=True)
    print("$" * 101)
    print("Plotting distributions of categorical columns...", flush=True)
    plot_start = time.time()
    for categorical_column in categorical_columns:
        print(f"\tPlotting {categorical_column} distribution")
        plot_catcol_distribution(df, categorical_column, save_path=savepath, save=save)
        plot_catcol_evolution_day(df, categorical_column, save_path=savepath, save=save)

    print("Plotting distributions of categorical columns by other categorical columns...", flush=True)
    for catcol1 in categorical_columns:
        for catcol2 in categorical_columns:
            if catcol1 != catcol2:
                print(f"\tPlotting {catcol1} distribution by {catcol2}")
                plot_catcat_distribution(df, catcol1, catcol2, save_path=savepath, save=save)

    print("Plotting distributions of numerical columns...", flush=True)
    for numerical_column in numerical_columns:
        print(f"\tPlotting {numerical_column} distribution", flush=True)
        unit = orbital_units[numerical_column]
        log_scale = numerical_column in log_scale_columns
        if numerical_column == 'INCLINATION':
            bins = 7
        elif numerical_column in ['MEAN_MOTION', 'SEMIMAJOR_AXIS', 'PERIOD', 'APOAPSIS']:
            bins = 5
        else:
            bins = 30
        plot_numcol_histogram(df, numerical_column, bins=bins, unit=unit, log_scale=log_scale, save_path=savepath,
                              save=save)
        if numerical_column not in ['MEAN_MOTION_DOT', 'BSTAR']:
            plot_numcol_violinplot(df, numerical_column, unit=unit, log_scale=log_scale, nodes_df=nodes_df,
                                   save_path=savepath, save=save)

    print("Plotting distributions of numerical columns by categorical columns...", flush=True)
    for numerical_column in numerical_columns:
        if numerical_column not in ['MEAN_MOTION_DOT', 'BSTAR']:
            unit = orbital_units[numerical_column]
            log_scale = numerical_column in log_scale_columns
            for categorical_column in categorical_columns:
                print(f"\tPlotting {numerical_column} distribution by {categorical_column}", flush=True)
                plot_numcat_violinplot(df, numerical_column, categorical_column, unit=unit, log_scale=log_scale,
                                       nodes_df=nodes_df, save_path=savepath, save=save)

    print("Plotting scatter plots of numerical columns...", flush=True)
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            x_column = numerical_columns[i]
            y_column = numerical_columns[j]
            for categorical_column in categorical_columns:
                print(f"\tPlotting {x_column} vs {y_column} by {categorical_column}", flush=True)
                plot_numnumcat_scatter(df, x_column, y_column, categorical_column, x_unit=orbital_units[x_column],
                                       y_unit=orbital_units[y_column], x_log_scale=x_column in log_scale_columns,
                                       y_log_scale=y_column in log_scale_columns, save_path=savepath, save=save)

    print("Plotting evolution of numerical columns per day for selected satellites...", flush=True)
    selected_sat_ids = filter_valid_satellites(df, num_satellites=num_satellites)
    nodes_df_copy = nodes_df.reset_index()
    for numerical_column in numerical_columns:
        print(f"\tPlotting {numerical_column} evolution per day for selected satellites", flush=True)
        unit = orbital_units[numerical_column]
        log_scale = numerical_column in log_scale_columns
        plot_numcol_evolution_day(df, numerical_column, selected_sat_ids, unit=unit,
                                  log_scale=log_scale, save_path=f'{savepath}orginal_', save=save)
        plot_numcol_evolution_day(nodes_df_copy, numerical_column, selected_sat_ids, unit=unit,
                                  log_scale=log_scale, save_path=f'{savepath}propagated_', save=save)

    print("Plotting correlation matrix of orbital parameters...", flush=True)
    plot_corr_heatmap(df, save_path=savepath, save=save)
    plot_end = time.time()

    print("$" * 101)
    load_time = (load_end - load_start) / 60
    plot_time = (plot_end - plot_start) / 60
    print(f"Time taken to load the data: {load_time:.2f} minutes", flush=True)
    print(f"Time taken to plot the data: {plot_time:.2f} minutes", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
