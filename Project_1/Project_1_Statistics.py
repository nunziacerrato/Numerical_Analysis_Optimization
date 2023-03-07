''' In this program we compute the scatterplot of the characteristic values of the growth factor
    and the relative backward error related to the LU factorization, i.e. the minimum value,
    the maximum value, the median value, and the standard deviation, as functions of the dimension
    of the input matrix. Moreover, we also obtain summary boxplots for these quantities, considering
    all the types of matrices investigated and varying their dimension.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Project_1 import *

# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 25
suptitle_size = 35
title_size = 22
legend_size = 18


# Extract the types of matrices considered
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
std_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])

mean_rel_back_err = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
min_rel_back_err = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
max_rel_back_err = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
std_rel_back_err = pd.DataFrame(columns = types_matrices,
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


    # For each matrix type considered add the mean value of the growth factor and the
    # relative backward error in the corresponding Data Frame
    # (note that indices in the Data Frame represent the dimension of the considered matrix)
    for matrix_type in types_matrices:
        
        mean_growth_factor.at[dim_matr,matrix_type] = np.mean(df_growth_factor[matrix_type])
        min_growth_factor.at[dim_matr,matrix_type] = np.min(df_growth_factor[matrix_type])
        max_growth_factor.at[dim_matr,matrix_type] = np.max(df_growth_factor[matrix_type])
        std_growth_factor.at[dim_matr,matrix_type] = np.std(df_growth_factor[matrix_type])

        mean_rel_back_err.at[dim_matr,matrix_type] = np.mean(df_rel_back_err[matrix_type])
        min_rel_back_err.at[dim_matr,matrix_type] = np.min(df_rel_back_err[matrix_type])
        max_rel_back_err.at[dim_matr,matrix_type] = np.max(df_rel_back_err[matrix_type])
        std_rel_back_err.at[dim_matr,matrix_type] = np.std(df_rel_back_err[matrix_type])


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
    ax_scatter[0].scatter(std_growth_factor.index,
                          np.log10(std_growth_factor[matrix_type].astype('float64')),
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
    ax_scatter[1].scatter(std_rel_back_err.index, 
                          np.log10(std_rel_back_err[matrix_type].astype('float64')),
                          label = r'$std(\delta)$')
    ax_scatter[1].tick_params(labelsize = tickparams_size)
    ax_scatter[1].set_xlabel('N', fontsize = xylabel_size)
    ax_scatter[1].set_ylabel(r'$\log_{10}(\delta)$', fontsize = xylabel_size)
    ax_scatter[1].set_title(f'Characteristic values of the relative backward error',
                            fontsize = title_size)
    ax_scatter[1].legend(fontsize = legend_size)
    fig_scatter.savefig(f'{common_path}_latex\\Plot\\Scatterplot_charact_values_for_{matrix_type}_matrices')
