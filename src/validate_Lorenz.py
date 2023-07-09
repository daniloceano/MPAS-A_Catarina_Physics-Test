# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    validate_Lorenz.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/12/30 13:23:07 by Danilo            #+#    #+#              #
#    Updated: 2023/07/09 16:39:15 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.metrics import mean_squared_error
import itertools
import seaborn as sns
import cmocean.cm as cmo
from matplotlib.colors import LinearSegmentedColormap

# Constants
terms = ['Az', 'Ae', 'Kz', 'Ke', 'Cz', 'Ca', 'Ck', 'Ce', 'BAz', 'BAe', 'BKz', 'BKe', 'Gz', 'Ge', '∂Az/∂t (finite diff.)', '∂Ae/∂t (finite diff.)', '∂Kz/∂t (finite diff.)','∂Ke/∂t (finite diff.)', 'RGz', 'RGe', 'RKz', 'RKe']
energy_terms = ['Az', 'Ae', 'Kz', 'Ke']
budget_diff_terms = ['∂Az/∂t (finite diff.)', '∂Ae/∂t (finite diff.)', '∂Kz/∂t (finite diff.)', '∂Ke/∂t (finite diff.)']
budget_diff_renamed = ['∂Az/∂t', '∂Ae/∂t', '∂Kz/∂t', '∂Ke/∂t']
main_terms = ['Az', 'Ae', 'Kz', 'Ke', 'Ca', 'Ce', 'Ck', 'Ge']
DiscreteColors = ['#58A8D6', '#74D669', '#D6BF5A', '#D6713A', '#D63631']

def get_exp_name(experiment):
    """
    Returns the name of the experiment based on the given experiment path.

    Parameters:
        experiment (str): The path of the experiment.

    Returns:
        str: The name of the experiment.
    """
    expname = os.path.basename(experiment)

    microp_options = ["thompson", "kessler", "wsm6", "off"]
    microp = next((option for option in microp_options if option in expname), None)
    if microp is None:
        raise ValueError("Microp option not found in experiment name.")

    cumulus_options = ["ntiedtke", "tiedtke", "freitas", "fritsch", "off"]
    cumulus = next((option for option in cumulus_options if option in expname), None)
    if cumulus is None:
        raise ValueError("Cumulus option not found in experiment name.")

    pbl_options = ["ysu", "mynn"]
    pbl = next((option for option in pbl_options if option in expname), None)

    return f"{microp}_{cumulus}_{pbl}" if pbl else f"{microp}_{cumulus}"


def process_data(results_directory, experiments):
    """
    Process the data from the given results directory and experiments.

    Parameters:
    - results_directory (str): The directory where the results are stored.
    - experiments (list): A list of experiment paths.

    Returns:
    - dictionary_df_experiments (dict): A dictionary where the keys are experiment paths and the values are DataFrames.
    - results_ERA (DataFrame): The processed data from the ERA5 dataset.
    """
    dictionary_df_experiments = {}
    for experiment in experiments:
        if len(experiment.split('_')) == 3:
            microp, cumulus, pbl = experiment.split('_')
            pattern = f"*{microp}*{cumulus}*{pbl}*"
            outfile = glob.glob(f'{results_directory}/{pattern}/{pattern}*csv')[0]
        elif len(experiment.split('_')) == 2:
            microp, cumulus = experiment.split('_')
            pattern = f"*{microp}*{cumulus}*"
            outfile = glob.glob(f'{results_directory}/{pattern}/{pattern}*csv')[0]

        if microp == 'off': continue
        
        df_exp = pd.read_csv(outfile, index_col=[0])
        df_exp['Datetime'] = pd.to_datetime(df_exp.Date) + pd.to_timedelta(
            df_exp.Hour, unit='h')
        time = df_exp.Datetime  
        df_exp.index = time
        df_exp = df_exp.drop(columns=['Datetime'])
        dictionary_df_experiments[experiment] = df_exp

    results_ERA = pd.read_csv(glob.glob(f'{results_directory}/*ERA5*/*ERA5*.csv')[0])
        
    results_ERA['Datetime'] = pd.to_datetime(results_ERA['Date']) + pd.to_timedelta(results_ERA['Hour'], unit='H')
    mask = (results_ERA['Datetime'] >= df_exp.index[0]) & (results_ERA['Datetime'] <= df_exp.index[-1])
    results_ERA = results_ERA.loc[mask]
    
    return dictionary_df_experiments, results_ERA

def calculate_metrics(dictionary_df_experiments, results_ERA, terms):
    """
    Calculate the root mean squared error (RMSE) and correlation coefficient for each term in the given experiments.
    
    Args:
        dictionary_df_experiments (dict): A dictionary containing dataframes for each experiment.
        results_ERA (dict): A dictionary containing the ERA results for each term.
        terms (list): A list of terms for which to calculate the metrics.
    
    Returns:
        df_rmse (pd.DataFrame): A dataframe containing the RMSE values for each term and experiment.
        df_corrcoef (pd.DataFrame): A dataframe containing the correlation coefficients for each term and experiment.
    """
    rmse_dict, corrcoef_dict = {}, {}
    for exp, df_exp in dictionary_df_experiments.items():
        rmse = {}
        corrcoef = {}
        for term in terms:
            reference = results_ERA[term].values.ravel()
            predicted = df_exp[term].values.ravel()
            rmse[term] =  np.sqrt(mean_squared_error(reference, predicted))
            corrcoef[term] = np.corrcoef(reference, predicted)[0, 1]
        rmse_dict[exp] = pd.DataFrame.from_dict([rmse])
        corrcoef_dict[exp] = pd.DataFrame.from_dict([corrcoef])
        
    df_rmse = pd.concat(rmse_dict)
    df_rmse.index = list(rmse_dict.keys())
        
    df_corrcoef = pd.concat(corrcoef_dict)
    df_corrcoef.index = list(corrcoef_dict.keys())
    
    return df_rmse, df_corrcoef


def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())

def sns_heatmap(data, title, figures_directory, benchmarks):
    """
    Generate a heatmap plot using seaborn.

    Args:
        data (array-like): The data to be plotted.
        title (str): The title of the plot.
        figures_directory (str): The directory where the plot will be saved.
        benchmarks (list): A list of benchmark values.

    Returns:
        None
    """
    cmap = None
    if 'RMSE_normalised_main' in title:
        cmap = LinearSegmentedColormap.from_list('Custom',
                                    DiscreteColors, len(DiscreteColors))
    elif 'R_Normalised' in title:
        cmap = LinearSegmentedColormap.from_list('Custom',
                                    DiscreteColors[::-1], len(DiscreteColors))
    else:
        if 'RMSE' in title:
            cmap = cmo.matter
        else:
            cmap = cmo.matter_r

    plt.close('all')
    if '48h' in benchmarks:
        y = 6
    elif benchmarks in ['72h_sst', '2403-2903']:
        y = 2
    else:
        y = 10
    f, ax = plt.subplots(figsize=(10, y))
    sns.heatmap(data, annot=False, linewidths=.5, ax=ax, cmap=cmap)

    plt.title(title)
    plt.tight_layout()
    stats_figure_directory = f'{figures_directory}/stats_Lorenz/'
    if not os.path.exists(stats_figure_directory):
            os.makedirs(stats_figure_directory)
    f.savefig(f'{stats_figure_directory}/{title}.png')
    print(f'Heatmap created: {stats_figure_directory}/{title}.png')

def main(results_directory):

    experiment_directory = ('/').join(os.path.dirname(results_directory).split('/')[:-1])
    benchmarks = os.path.basename(os.path.dirname(results_directory)).split('LEC_Results_')[1]

    stats_directory = os.path.join(experiment_directory, f'stats_{benchmarks}')
    figures_directory = os.path.join(experiment_directory, f'Figures_{benchmarks}')
    lec_directory = os.path.join(experiment_directory, f'LEC_Results_{benchmarks}')

    results = glob.glob(f'{lec_directory}/*')

    experiments = []
    for experiment in results:
        if 'ERA5' not in experiment:
            experiment_name = get_exp_name(experiment)
            experiments.append(experiment_name)

    # Process data
    dictionary_df_experiments, results_ERA = process_data(results_directory, experiments)

    # Calculate metrics for all LEC terms
    df_rmse_all, df_corrcoef_all = calculate_metrics(dictionary_df_experiments, results_ERA, terms)
    df_rmse_all, df_corrcoef_all = df_rmse_all.sort_index(ascending=True), df_corrcoef_all.sort_index(ascending=True)
    
    # Differentiate RMSE between energy and not energy
    df_rmse_energy = df_rmse_all[energy_terms]
    df_rmse_not_energy = df_rmse_all.drop(columns=energy_terms)
    
    # RMSE for only the main terms
    df_rmse_main = df_rmse_all[main_terms]

    # Normalize values for comparison
    df_rmse_main_norm = normalize_df(df_rmse_main)
    df_corrcoef_norm = normalize_df(df_corrcoef_all)

    # Visualize heatmaps
    sns_heatmap(df_rmse_energy, 'RMSE_energy', figures_directory, benchmarks)
    sns_heatmap(df_rmse_not_energy, 'RMSE_not-energy', figures_directory, benchmarks)
    sns_heatmap(df_rmse_main_norm, 'RMSE_normalised_main', figures_directory, benchmarks)
    sns_heatmap(df_corrcoef_all, 'R', figures_directory, benchmarks)
    sns_heatmap(df_corrcoef_norm, 'R_normalised', figures_directory, benchmarks)

    # Save RMSE for main terms,normalised, for score table
    df_rmse_main_norm.to_csv(f'{stats_directory}/Lorenz-main_RMSE_normalised.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", nargs="?", default='../experiments_48h/LEC_Results_48h/',
                        help="Path to the LEC results directory")
    args = parser.parse_args()
    main(args.results_directory)