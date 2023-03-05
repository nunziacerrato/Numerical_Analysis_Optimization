''' This program computes the distributions of the growth factor and the relative backward error 
    for all the types of matrices considered and the relationship between the characteristic values
    of such distributions and the dimension of the matrices.'''

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

Histograms = False

# dim_matr_max = 20

# Extract the types of matrices considered
# types_matrices = [key for key in create_dataset(1,2).keys()]
types_matrices = create_dataset(1,2).keys()

# Create the Data Frames to store the mean, min, max value of the growth factor and 
# the relative backward error for all the different types of matrices considered: columns
# represent the type of matrix, indeces represent the progressiove dimension of the matrices.
mean_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
min_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
max_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
var_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])

mean_rel_back_err = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
min_rel_back_err = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
max_rel_back_err = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
var_rel_back_err = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])

# Cycle on the dimension of the input matrices
for dim_matr in range(2,dim_matr_max+1):

    # Access the data: growth factor and relative backward error for all the considered types of
    # matrices of a given dimension - data read from Excel files and stored in DataFrames
    df_growth_factor = pd.read_excel(f'{common_path}\\Data\\'
                                     f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                     sheet_name = 'growth_factor')
    
    df_rel_back_err = pd.read_excel(f'{common_path}\\Data\\'
                                    f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                    sheet_name = 'rel_back_err')
    
    if Histograms == True:
        # Create the figures for the histograms
        fig_hist_g, ax_hist_g = plt.subplots(figsize=(15,10), nrows = 2, ncols = 3)
        fig_hist_g.suptitle('Distributions of the growth factor - '\
                            fr'$({dim_matr}\times{dim_matr})$ matrices', fontsize = suptitle_size)
    
        fig_hist_rbe, ax_hist_rbe = plt.subplots(figsize=(15,10), nrows = 2, ncols = 3)
        fig_hist_rbe.suptitle('Distributions of the relative backward error - '\
                              fr'$({dim_matr}\times{dim_matr})$ matrices', fontsize = suptitle_size)

    # Create boxplot to obtain a summary of the caracteristic values of the growth factor and the
    # relative backward error considering as fixed the dimension of the matrices - useful for
    # comparing the different types of matrices taken into consideration (ouliers discarded).
    fig_boxplot_matrices, ax_boxplot_matrices = plt.subplots(figsize=(22,10), nrows = 1, ncols = 2, constrained_layout = True)
    fig_boxplot_matrices.suptitle(fr'Matrices of dimension $({dim_matr}\times{dim_matr})$', fontsize = 25)
    ax_boxplot_matrices[0].boxplot(df_growth_factor, labels = df_growth_factor.columns,
                                   showfliers=False)
    ax_boxplot_matrices[0].set_title('Growth factor', fontsize = title_size)
    ax_boxplot_matrices[0].tick_params(labelsize = tickparams_size)
    ax_boxplot_matrices[1].boxplot(df_rel_back_err, labels = df_rel_back_err.columns,
                                   showfliers=False)
    ax_boxplot_matrices[1].set_title('Relative backward error', fontsize = title_size)
    ax_boxplot_matrices[1].tick_params(labelsize = tickparams_size)
    fig_boxplot_matrices.savefig(f'{common_path}_latex\\Plot\\Boxplot_dim_matr={dim_matr}')
    # plt.show()

    # For each matrix type considered add the mean value of the growth factor and the
    # relative backward error in the corresponding Data Frame
    # (note that indices in the Data Frame represent the dimension of the considered matrix)
    for matrix_type in types_matrices:

        if Histograms == True:

            index = list(df_growth_factor.columns).index(matrix_type)

            # Distributions of the growth factor
            ax_hist_g[index//3,index%3].hist(df_growth_factor[matrix_type], bins = 'auto',\
                                        histtype='step', fill = False, label = matrix_type, log = True)
            ax_hist_g[index//3,index%3].set_xlabel(fr'$\gamma$', fontsize = xylabel_size)
            ax_hist_g[index//3,index%3].set_ylabel(r'$\log_{10}(f)$', fontsize = xylabel_size)
            ax_hist_g[index//3,index%3].legend(fontsize = legend_size)

            # for ax in ax_hist_g.flat:
            #     ax.label_outer()
            #     for tk in ax.get_xticklabels():
            #         tk.set_visible(True)
            # ax_hist_g.xaxis.set_tick_params(labelbottom=True)

            # Distributions of the relative backward error
            ax_hist_rbe[index//3,index%3].hist(df_rel_back_err[matrix_type], bins = 'auto',\
                                        histtype='step', fill = False, label = matrix_type, log = True)
            ax_hist_rbe[index//3,index%3].set_xlabel(fr'$\delta$', fontsize = xylabel_size)
            ax_hist_rbe[index//3,index%3].set_ylabel(r'$\log_{10}(f)$', fontsize = xylabel_size)
            ax_hist_rbe[index//3,index%3].legend(fontsize = legend_size)

            # for ax in ax_hist_rbe.flat:
            #     ax.label_outer()
        
        mean_growth_factor.at[dim_matr,matrix_type] = np.mean(df_growth_factor[matrix_type])
        min_growth_factor.at[dim_matr,matrix_type] = np.min(df_growth_factor[matrix_type])
        max_growth_factor.at[dim_matr,matrix_type] = np.max(df_growth_factor[matrix_type])
        var_growth_factor.at[dim_matr,matrix_type] = np.std(df_growth_factor[matrix_type])

        mean_rel_back_err.at[dim_matr,matrix_type] = np.mean(df_rel_back_err[matrix_type])
        min_rel_back_err.at[dim_matr,matrix_type] = np.min(df_rel_back_err[matrix_type])
        max_rel_back_err.at[dim_matr,matrix_type] = np.max(df_rel_back_err[matrix_type])
        var_rel_back_err.at[dim_matr,matrix_type] = np.std(df_rel_back_err[matrix_type])


# Scatterplot of the characteristic values of the growth factor and the relative backward error
# cycling on the types of matrices considered.
for matrix_type in types_matrices:
    fig_scatter, ax_scatter= plt.subplots(figsize=(18,10), nrows = 1, ncols = 2, constrained_layout = True)
    fig_scatter.suptitle(f'{matrix_type} matrices', fontsize = suptitle_size)

    # Growth factor
    ax_scatter[0].scatter(mean_growth_factor.index,
                          np.log10(mean_growth_factor[matrix_type].astype('float64')),
                          label = r'mean($\gamma$)')
    ax_scatter[0].scatter(min_growth_factor.index,
                          np.log10(min_growth_factor[matrix_type].astype('float64')),
                          label = r'min($\gamma$)')
    ax_scatter[0].scatter(max_growth_factor.index,
                          np.log10(max_growth_factor[matrix_type].astype('float64')),
                          label = r'max($\gamma$)')
    ax_scatter[0].scatter(var_growth_factor.index,
                          np.log10(var_growth_factor[matrix_type].astype('float64')),
                          label = r'std($\gamma$)')
    ax_scatter[0].tick_params(labelsize = tickparams_size)
    ax_scatter[0].set_xlabel('N', fontsize = xylabel_size)
    ax_scatter[0].set_ylabel(r'$\log_{10}(\gamma)$',fontsize = xylabel_size)
    ax_scatter[0].set_title(f'Characteristic values of the growth factor', fontsize = title_size)
    ax_scatter[0].legend(fontsize = legend_size)

    # Relative backward error
    ax_scatter[1].scatter(mean_rel_back_err.index, 
                          np.log10(mean_rel_back_err[matrix_type].astype('float64')), 
                          label = r'$mean(\delta)$')
    ax_scatter[1].scatter(min_rel_back_err.index,
                          np.log10(min_rel_back_err[matrix_type].astype('float64')),
                          label = r'$min(\delta)$')
    ax_scatter[1].scatter(max_rel_back_err.index,
                          np.log10(max_rel_back_err[matrix_type].astype('float64')),
                          label = r'$max(\delta)$')
    ax_scatter[1].scatter(var_rel_back_err.index, 
                          np.log10(var_rel_back_err[matrix_type].astype('float64')),
                          label = r'$var(\delta)$')
    ax_scatter[1].tick_params(labelsize = tickparams_size)
    ax_scatter[1].set_xlabel('N', fontsize = xylabel_size)
    ax_scatter[1].set_ylabel(r'$\log_{10}(\delta)$', fontsize = xylabel_size)
    ax_scatter[1].set_title(f'Characteristic values of the relative backward error',
                            fontsize = title_size)
    ax_scatter[1].legend(fontsize = legend_size)
    fig_scatter.savefig(f'{common_path}_latex\\Plot\\Scatterplot_charact_values_for_{matrix_type}_matrices')
    # plt.show()



# Plot Histograms
if False:

    types_matrices = create_dataset(1,2).keys()
    dim_matr_list = [2,25,50]
    num_matr = 500

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
            fig_hist.suptitle(f'{matrix_type}', fontsize = suptitle_size)

            # Plot of the distribution of the growth factor
            ax_hist[0].hist(df_growth_factor[matrix_type], bins = 'auto',\
                                        histtype='step', fill = False, label = matrix_type, log = True)
            ax_hist[0].set_xlabel(fr'$\gamma$', fontsize = xylabel_size)
            ax_hist[0].set_ylabel(r'$\log_{10}(f)$', fontsize = xylabel_size)
            ax_hist[0].set_title(f'Distribution of the growth factor')
            ax_hist[0].legend(fontsize = legend_size)

            # Plot of the distribution of the relative backward error
            ax_hist[1].hist(df_rel_back_err[matrix_type], bins = 'auto',\
                                        histtype='step', fill = False, label = matrix_type, log = True)
            ax_hist[1].set_xlabel(fr'$\gamma$', fontsize = xylabel_size)
            ax_hist[1].set_ylabel(r'$\log_{10}(f)$', fontsize = xylabel_size)
            ax_hist[1].set_title(f'Distribution of the relative backward error')
            ax_hist[1].legend(fontsize = legend_size)

            



if False:
    fig_scatter_mean_g, ax_scatter_mean_g = plt.subplots(figsize=(15,10), nrows = 2, ncols = 3)
    fig_scatter_mean_g.suptitle(f'Mean value of the growth factor with respect to the dimension of the matrices')

    # fig_boxplot_g, ax_boxplot_g = plt.subplots(figsize=(15,10), nrows = 2, ncols = 3)
    # fig_boxplot_g.suptitle(f'Box plot of the growth factor with respect to the dimension of the matrices')

    for dim_matr in range(2,dim_matr_max+1):

        # Access the data: growth factor and relative backward error
        df_growth_factor = pd.read_excel(f'{common_path}\\Data\\'
                                        f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                        sheet_name = 'growth_factor')
        df_rel_back_err = pd.read_excel(f'{common_path}\\Data\\'
                                        f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                        sheet_name = 'rel_back_err')

        # # Create the figures for the histograms
        # fig_hist_g, ax_hist_g = plt.subplots(figsize=(15,10), nrows = 2, ncols = 3)
        # fig_hist_g.suptitle(fr'Distributions of the growth factor - $({dim_matr}\times{dim_matr})$ matrices')

        # fig_hist_rbe, ax_hist_rbe = plt.subplots(figsize=(15,10), nrows = 2, ncols = 3)
        # fig_hist_rbe.suptitle(fr'Distributions of the relative backward error - $({dim_matr}\times{dim_matr})$ matrices')

        # fig_boxplot_matrices, ax_boxplot_matrices = plt.subplots()
        # ax_boxplot_matrices.boxplot(df_growth_factor, labels = df_growth_factor.columns)
        # # plt.show()

        # Plot the histograms in a grid
        for matrix_type in df_growth_factor.columns:
            index = list(df_growth_factor.columns).index(matrix_type)

            # Scatterplot of the growth factor with respect to the dimension of the matrices
            ax_scatter_mean_g[index//3,index%3].scatter(dim_matr, np.mean(df_growth_factor[matrix_type]),
                                                        color = 'blue')
            ax_scatter_mean_g[index//3,index%3].set_title(matrix_type)

            # Box plot of the growth factor with respect to the dimension of the matrices
            # boxplot = df_growth_factor[matrix_type].boxplot(columns = f'{dim_matr}', return_type = 'axes')
            # ax_boxplot_g[index//3,index%3].boxplot(df_growth_factor[matrix_type], labels = f'{dim_matr}')
            # ax_boxplot_g[index//3,index%3].set_title(matrix_type)

            # # Distributions of the growth factor
            # ax_hist_g[index//3,index%3].hist(df_growth_factor[matrix_type], bins = 'auto',\
            #                             histtype='step', fill = False, density = True, label = matrix_type)
            # ax_hist_g[index//3,index%3].legend()

            # # Distributions of the relative backward error
            # ax_hist_rbe[index//3,index%3].hist(df_rel_back_err[matrix_type], bins = 'auto',\
            #                             histtype='step', fill = False, density = True, label = matrix_type)
            # ax_hist_rbe[index//3,index%3].legend()

    plt.show()
