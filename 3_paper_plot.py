import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde


def plot_metric_distributions(data_path, output_dir):
    # Read CSV file
    interaction_index_info = pd.read_csv(data_path)

    # Calculate the number of track_id in each row
    interaction_index_info['track_count'] = interaction_index_info['track_id'].apply(lambda x: len(x.split(';')))

    # Calculate the duration with conditional division based on dataset
    interaction_index_info['duration'] = interaction_index_info.apply(
        lambda row: (row['end'] - row['start']) / 20 if row['dataset'] == 'nuplan_train' else (row['end'] - row['start']) / 10,
        axis=1
    )

    # Combine 'interaction_single' and 'interaction_multi' into 'interaction'
    interaction_index_info['dataset'] = interaction_index_info['dataset'].replace({'interaction_single': 'interaction', 'interaction_multi': 'interaction'})

    # Metrics to plot
    metrics = ['intensity', 'duration']

    # Get unique datasets
    datasets = interaction_index_info['dataset'].unique()

    # Set specific colors for each dataset
    custom_colors = {
        'waymo_train': '#025436',
        'nuplan_train': '#BC5565',
        'lyft_train_full': '#3868A6',
        'interaction_single': '#F7D176',
        'interaction_multi': '#F7D176'
    }

    # Dynamically generate color mapping for datasets
    color_palette = sns.color_palette("husl", len(datasets))
    dataset_colors = {}

    for i, dataset in enumerate(datasets):
        if dataset not in custom_colors:  # If dataset is not in predefined custom colors, assign from palette
            dataset_colors[dataset] = color_palette[i % len(color_palette)]
        else:
            dataset_colors[dataset] = custom_colors[dataset]

    # Plot the distribution of each metric
    for i, metric in enumerate(metrics, 1):
        plt.figure(figsize=(10, 6))

        for j, dataset in enumerate(datasets):
            subset = interaction_index_info[interaction_index_info['dataset'] == dataset][metric]
            
            if metric == 'track_count':
                # Count the frequency of each integer value between 2 and 7
                value_counts = subset.value_counts().sort_index()
                x_values = np.arange(2, 8)  # Limit x-axis to 2 to 7
                y_values = [value_counts.get(x, 0) for x in x_values]
                
                plt.bar(x_values, y_values, alpha=0.7, label=dataset, color=dataset_colors[dataset], edgecolor='black')
            elif metric == 'intensity':
                # Apply KDE for intensity and convert to frequency by multiplying by total number of samples
                kde = gaussian_kde(subset)
                x_range = np.linspace(subset.min(), subset.max(), 1000)
                
                # Multiply KDE values by the total number of data points to get the frequency (count)
                kde_values = kde(x_range) * len(subset)
                
                plt.plot(x_range, kde_values, label=dataset, color=dataset_colors[dataset], lw=2, alpha=0.7)

            else:
                # Use KDE for continuous metrics
                kde = gaussian_kde(subset)
                x_range = np.linspace(subset.min(), subset.max(), 1000)
                plt.plot(x_range, kde(x_range), label=dataset, color=dataset_colors[dataset], alpha=0.7)

        plt.title(f'Distribution of {metric} by dataset')
        plt.xlabel(metric)
        plt.ylabel('Frequency' if metric != 'track_count' else 'Density')
        plt.legend()

        if metric == 'duration':
            plt.xlim(0, 4.5)  # Adjust for duration

        # Save the picture
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{metric}_distribution.png')
        plt.savefig(output_file, dpi=300)
        plt.close()


if __name__ == "__main__":
    # set the default data path and output directory
    data_path = 'data/3_paperplot_data/all_results.csv'
    output_dir = 'figs/paper_plot'
    
    # if the output directory does not exist, create it
    os.makedirs(output_dir, exist_ok=True)
    
    # call the plotting function
    plot_metric_distributions(data_path, output_dir)
