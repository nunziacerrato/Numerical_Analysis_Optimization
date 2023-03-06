''' In this program we obtain the distributions of the growth factor and the relative backward error
    related to the LU factorization for all the types of matrices investigated and dimensions fixed
    to the values contained in the list "dim_matr_list", defined in the beginning of the code.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Project_1 import *

# Set size parameters for the plots
tickparams_size = 15
xylabel_size = 16
suptitle_size = 25
title_size = 18
legend_size = 15

# Set matrix parameters for the plot
types_matrices = create_dataset(1,2).keys()
dim_matr_list = [25]

# Cycle on the dimension of the input matrices
for dim_matr in dim_matr_list:

    # Access the data: growth factor and relative backward error for all the considered types of
    # matrices of a given dimension - data read from Excel files and stored in DataFrames
    df_growth_factor = pd.read_excel(f'{common_path}\\Data\\'
                                    f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                    sheet_name = 'growth_factor')
    
    df_rel_back_err = pd.read_excel(f'{common_path}\\Data\\'
                                    f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                    sheet_name = 'rel_back_err')


    for matrix_type in types_matrices:
        # Create the figures for the histograms
        fig_hist, ax_hist = plt.subplots(figsize=(15,10), nrows = 1, ncols = 2)
        fig_hist.suptitle(f'{matrix_type} matrices', fontsize = suptitle_size)

        # Create the figures for the boxplots
        fig_boxplot, ax_boxplot = plt.subplots(figsize=(15,10), nrows = 1, ncols = 2)
        fig_boxplot.suptitle(f'{matrix_type} matrices', fontsize = suptitle_size)

        # Compute min, max, mean and standard deviation of the growth factor
        min_g = np.min(df_growth_factor[matrix_type])
        max_g = np.max(df_growth_factor[matrix_type])
        mean_g = np.mean(df_growth_factor[matrix_type])
        std_g = np.std(df_growth_factor[matrix_type])

        # Compute min, max, mean and standard deviation of the relative backward error
        min_rbe = np.min(df_rel_back_err[matrix_type])
        max_rbe = np.max(df_rel_back_err[matrix_type])
        mean_rbe = np.mean(df_rel_back_err[matrix_type])
        std_rbe = np.std(df_rel_back_err[matrix_type])

        label_g = fr'min($\gamma$)={np.round(min_g,2)},' + '\n' + \
                  fr'max($\gamma$)={np.round(max_g,2)},' + '\n' + \
                  fr'mean($\gamma$)={np.round(mean_g,2)},' + '\n' + \
                  fr'std($\gamma$)={np.round(std_g,2)}'
        label_rbe = fr'min($\delta$)={min_rbe:.2E},' + '\n' + \
                    fr'max($\delta$)={max_rbe:.2E},' + '\n' + \
                    fr'mean($\delta$)={mean_rbe:.2E},' + '\n' + \
                    fr'std($\delta$)={std_rbe:.2E}'

        # Plot of the distribution of the growth factor
        ax_hist[0].hist(df_growth_factor[matrix_type], bins = 'auto',\
                                    histtype='step', fill = False, label = label_g, log = True)
        ax_hist[0].set_xlabel(r'$\gamma$', fontsize = xylabel_size)
        ax_hist[0].set_ylabel('f', fontsize = xylabel_size)
        ax_hist[0].set_title(f'Distribution of the growth factor')
        ax_hist[0].set_xscale("log")
        ax_hist[0].legend(fontsize = legend_size)

        # Plot of the distribution of the relative backward error
        ax_hist[1].hist(df_rel_back_err[matrix_type], bins = 'auto',\
                                    histtype='step', fill = False, label = label_rbe, log = True)
        ax_hist[1].set_xlabel(fr'$\delta$', fontsize = xylabel_size)
        ax_hist[1].set_ylabel('f', fontsize = xylabel_size)
        ax_hist[1].set_title(f'Distribution of the relative backward error')
        ax_hist[1].set_xscale("log")
        ax_hist[1].legend(fontsize = legend_size)

        fig_hist.savefig(f'{common_path}_latex\\Plot\\Distributions_for_{matrix_type}_matrices_of_dim_{dim_matr}')

        if False:
            # Plot of the boxplots of the growth factor
            ax_boxplot[0].boxplot(df_growth_factor[matrix_type], showfliers=False)
            ax_boxplot[0].set_title(f'Boxplot of the growth factor')
            # ax_boxplot[0].legend(fontsize = legend_size)


            # Plot of the boxplots of the relative backward error
            ax_boxplot[1].boxplot(df_rel_back_err[matrix_type], showfliers=False)
            ax_boxplot[1].set_yscale("log")
            ax_boxplot[1].set_title(f'Boxplot of the relative backward error')
            # ax_boxplot[1].legend(fontsize = legend_size)

            fig_boxplot.savefig(f'{common_path}_latex\\Plot\\Boxplot_for_{matrix_type}_matrices_of_dim_{dim_matr}')
