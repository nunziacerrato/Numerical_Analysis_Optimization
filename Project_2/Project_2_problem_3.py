''' '''

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from Project_2 import *

# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 25
suptitle_size = 35
title_size = 22
legend_size = 18

if False:
    A = np.array([[-4.,-2.,-4.,-2.],[2.,-2.,2.,1.],[-800,200,-800,-401]])
    U, singular_values, V_transpose = np.linalg.svd(A, compute_uv=True)
    pseudoinverse = np.linalg.pinv(A)
    spectral_cond_num = np.linalg.cond(A)

    print(f'singular values of A = {singular_values}')
    print(f'pseudoinverse of A = {pseudoinverse}')
    print(f'spectral condition number of A = {spectral_cond_num}')


    A_rank_1 = singular_values[0]*np.outer(U[:,0],V_transpose[0])
    A_rank_2 = A_rank_1 + singular_values[1]*np.outer(U[:,1],V_transpose[1])
    spectral_cond_num_A_rank_2 = singular_values[0]/singular_values[1]
    print(f'A_rank_1 = {A_rank_1}')
    print(f'A_rank_2 = {A_rank_2}')
    print(f'Spectral condition number of A_rank_2 = {spectral_cond_num_A_rank_2}')

    A_diff_1 = A - A_rank_1
    Frob_norm_1 = np.linalg.norm(A_diff_1, ord = 'fro')
    Two_norm_1 = np.linalg.norm(A_diff_1, ord = None)
    print(f'Frobenious norm between A and A_rank_1 = {Frob_norm_1}')
    
    A_diff_2 = A - A_rank_2
    Frob_norm_2 = np.linalg.norm(A_diff_2, ord = 'fro')
    Two_norm_2 = np.linalg.norm(A_diff_2, ord = None)
    print(f'Frobenious norm between A and A_rank_2 = {Frob_norm_2}')


if False:
    theta = np.linspace(0,2*np.pi,num=10000)
    A = np.array([[1,2],[0,2]])
    x0 = np.array([np.cos((theta*180)/np.pi),np.sin((theta*180)/np.pi)])
    # x = np.array([np.cos((theta*180)/np.pi)+2*np.sin((theta*180)/np.pi),2*np.sin((theta*180)/np.pi)])
    y = A@x0
    plt.scatter(y[0],y[1], s=0.1)
    plt.title('Image of the unit circle', fontsize = title_size)
    plt.tick_params(labelsize = tickparams_size)
    plt.xlabel(r'$y_{0}$', fontsize = xylabel_size)
    plt.ylabel(r'$y_{1}$', fontsize = xylabel_size)
    plt.show()


if True:
    n_list = [10,20,50,100]
    precision = np.finfo(float).eps
    # Compute the matrix R
    def R_matrix(n):
        R = np.triu(-np.ones((n,n)),k=1) + np.eye(n)
        return R

    if False:
        for n in n_list:
            singular_values = np.linalg.svd(R_matrix(n), compute_uv=False)

            fig, ax = plt.subplots(figsize=(15,10))
            plt.scatter(range(1,n+1),np.log10(singular_values), label = r'$\sigma_{n}$')
            plt.hlines(np.log10(precision*singular_values[0]),1,n, color = 'red', label=r'$u\sigma_{1}$')
            plt.title(f'Singular values of R for n = {n}', fontsize = title_size)
            plt.tick_params(labelsize = tickparams_size)
            plt.xlabel('n', fontsize = xylabel_size)
            # plt.legend(fontsize = legend_size)
            # plt.show()
            fig.savefig(f'Project_2_latex\\Plot\\Singular_values_of_R_for_n={n}')


    for n in range(1,n_list[-1]+1):
        R = R_matrix(n)
        cond_num = np.linalg.cond(R)

        plt.scatter(n,np.log10(cond_num), color = 'blue')
    plt.title('Spectral condition number of R as a function of its dimension', fontsize = title_size)
    plt.ylim(-1,22)
    plt.vlines(50,-1,22, linestyle = 'dotted', color = 'gray', label = 'n=50')
    plt.vlines(60,-1,22, linestyle = 'dotted', color = 'black', label = 'n=60')
    plt.tick_params(labelsize = tickparams_size)
    plt.xlabel('n', fontsize = xylabel_size)
    plt.ylabel(r'$k_{2}(R_{n})$', fontsize = xylabel_size)
    plt.legend(fontsize = legend_size)
    plt.show()
        
    
