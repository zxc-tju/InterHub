import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde


def plot_metric_distributions(data_path, output_dir):
    # read csv file
    interaction_index_info = pd.read_csv(data_path)

    # calculate the number of track_id in each row
    interaction_index_info['track_count'] = interaction_index_info['track_id'].apply(lambda x: len(x.split(';')))

    # calculate the duration
    interaction_index_info['duration'] = interaction_index_info['end'] - interaction_index_info['start']

    # metrics to plot
    metrics = ['intensity', 'PET', 'duration', 'track_count']

    # get unique datasets
    datasets = interaction_index_info['dataset'].unique()

    # set colors for each dataset
    colors = sns.color_palette("husl", len(datasets))

    # plot the distribution of each metric
    for i, metric in enumerate(metrics, 1):
        plt.figure(figsize=(10, 6))
        
        for j, dataset in enumerate(datasets):
            subset = interaction_index_info[interaction_index_info['dataset'] == dataset][metric]
            
            if metric == 'track_count':
                # count the frequency of each integer value between 2 and 7
                value_counts = subset.value_counts().sort_index()
                x_values = np.arange(2, 8)  # limit x-axis to 2 to 7
                y_values = [value_counts.get(x, 0) for x in x_values]
                
                plt.bar(x_values, y_values, alpha=0.7, label=dataset, color=colors[j % len(colors)], edgecolor='black')
            else:

                kde = gaussian_kde(subset)
                x_range = np.linspace(subset.min(), subset.max(), 1000)
                plt.plot(x_range, kde(x_range), label=dataset, color=colors[j % len(colors)])
        
        plt.title(f'Distribution of {metric} by dataset')
        plt.xlabel(metric)
        plt.ylabel('Density' if metric != 'track_count' else 'Frequency')
        plt.legend()
        
        # save the picture
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{metric}_distribution.png')
        plt.savefig(output_file, dpi=300)
        plt.close()


if __name__ == "__main__":
    # set the default data path and output directory
    data_path = 'dataset/results.csv'
    output_dir = 'visualize/paper_plot'
    
    # if the output directory does not exist, create it
    os.makedirs(output_dir, exist_ok=True)
    
    # call the plotting function
    plot_metric_distributions(data_path, output_dir)
