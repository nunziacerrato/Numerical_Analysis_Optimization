''' This program computes the distributions of the growth factor and the relative backward error 
    for all the types of matrices considered and the relationship between the characteristic values
    of such distributions and the dimension of the matrices.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Project_1_2 import *

num_matr = 500
dim_matr_max = 6

common_path = "C:\\Users\\cerra\\Documents\\GitHub\\Numerical_Analysis_Optimization\\Project_1"

# Extract the types of matrices considered
types_matrices = [key for key in create_dataset(1,2).keys()]

# Create the Data Frames to store the mean, min, max value of the growth factor for all the
# different types of matrices considered: columns represent the type of matrix,
# indeces represent the progressiove dimension of the matrices.
mean_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
min_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
max_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])
var_growth_factor = pd.DataFrame(columns = types_matrices,
                                  index = [dim for dim in range(2,dim_matr_max)])

# Cycle on dimension of input matrices
for dim_matr in range(2,dim_matr_max+1):

    # Access the data: growth factor and relative backward error for all the considered types of
    # matrices of a given dimension - data read from Excel files and stored in DataFrames
    df_growth_factor = pd.read_excel(f'{common_path}\\Data\\'
                                     f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                     sheet_name = 'growth_factor')
    
    df_rel_back_err = pd.read_excel(f'{common_path}\\Data\\'
                                    f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx',
                                    sheet_name = 'rel_back_err')

    # For each matrix type considered add the mean value fo the growth factor in the Data Frame
    # (note that indices in the Data Frame represent the dimension of the considered matrix)
    for matrix_type in types_matrices:
        mean_growth_factor.at[dim_matr,matrix_type] = np.mean(df_growth_factor[matrix_type])
        min_growth_factor.at[dim_matr,matrix_type] = np.min(df_growth_factor[matrix_type])
        max_growth_factor.at[dim_matr,matrix_type] = np.max(df_growth_factor[matrix_type])
        var_growth_factor.at[dim_matr,matrix_type] = np.var(df_growth_factor[matrix_type])
# print(type(mean_growth_factor['Random']))
# mean_growth_factor['logarithm'] = np.log10(mean_growth_factor['Random'])
# print(mean_growth_factor['logarithm'])
 
plt.scatter(mean_growth_factor.index, np.log(mean_growth_factor['Random'].astype('float64')),
            label = r'mean($\gamma$)')
plt.scatter(min_growth_factor.index, np.log(min_growth_factor['Random'].astype('float64')),
            label = r'min($\gamma$)')
plt.scatter(max_growth_factor.index, np.log(max_growth_factor['Random'].astype('float64')),
            label = r'max($\gamma$)')
plt.scatter(var_growth_factor.index, np.log(var_growth_factor['Random'].astype('float64')),
            label = r'var($\gamma$)')

plt.legend()
plt.show()

    # Cycle on the matrix type
    # for matrix_type in df_growth_factor.columns:
    # #     index = list(df_growth_factor.columns).index(matrix_type)
    #     df_growth_factor_dim[matrix_type] = 

    #     # Scatterplot of the growth factor with respect to the dimension of the matrices
    #     ax_scatter_mean_g[index//3,index%3].scatter(dim_matr, np.mean(df_growth_factor[matrix_type]),
    #                                                 color = 'blue')
    #     ax_scatter_mean_g[index//3,index%3].set_title(matrix_type)


# print(df_growth_factor)


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
