import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Project_5 import *

func_b = lambda x : 2*x[0] - x[1]**2
grad_b = lambda x : np.array([2, -2*x[1]])
hess_b = lambda x : np.array([[0, 0],[0, -2]])

c_b = lambda x : np.array([1 - x[0]**2 - x[1]**2 , x[0], x[1]])
grad_c_b = lambda x : np.array([[-2*x[0],-2*x[1]],[1,0],[0,1]])
def hess_c_b(x):
    hess_c = np.zeros((2,3,2))
    hess_c[:,0,:] = np.array([[-2, 0],[0, -2]])
    return hess_c

method = 'basic'
mu = 1e-5
tol = 1e-12
seed = 14
common_path = "Project_5"

max_val = 10
start = 1e-12
step = 0.1
l_array = list(np.arange(start,max_val+1e-10,step))
z_array = list(np.arange(start,max_val+1e-10,step))
dim_l = len(l_array)
dim_z = len(z_array)
M = np.zeros([dim_l,dim_z])
tol_x = 1e-1

mu_list = [0,1e-12,1e-3]
x0_list = [np.array([0.1,0.1]),np.array([0.1,0.9]),np.array([0.9,0.1]),np.array([0.3,0.7]),
        np.array([0.7,0.3]),np.array([0.5,0.5]),np.array([0.25,0.25]),np.array([1,0])]

colorplot = False
x0_list = [np.array([0.1,0.9])]
###### COLORPLOT #######
if colorplot == True:
    curv = True
    for mu in mu_list:
        for x0 in x0_list:
            # print(f'x0 = {x0}')
            for l_int in l_array:
                for z_int in z_array:
                    l0 = l_int*np.ones(3)
                    z0 = z_int*np.ones(3)
                    print(f'x0 = {x0}, iter = {l_int,z_int}')
                    results = int_point(func_b, grad_b, hess_b, c_b, grad_c_b, hess_c_b, x0, method=method, alpha=1., beta=1.,
                                gamma=1., mu=mu, tol=tol, maxit=100, l0=l0, z0=z0, curv=curv, seed=seed)
                    k = results['n_iter']
                    min_val = results['x_min']
                    min_exact = np.array([0.,1.])
                    min_err = np.array([0.,0.])
                    if (abs(min_val-min_err)).all() < 1e-12:
                    # if np.linalg.norm(min_val-min_exact) > tol_x and np.linalg.norm(min_val-min_err) < tol_x:
                        k = 150
                    l_index = l_array.index(l_int)
                    z_index = z_array.index(z_int)
                    M[l_index,z_index] = k

            fig = plt.figure()
            X, Y = np.meshgrid(l_array, z_array)
            plt.pcolormesh(X, Y, M, cmap='jet')
            plt.title(fr'Convergence iteration - $\mu$ = {mu}, $x_{{0}}$={x0}')
            plt.xlabel(r'$\lambda$')
            plt.ylabel("z")
            plt.colorbar()
            if curv == True:
                fig.savefig(f'{common_path}_latex\\Plot\\func_b\\colorplot\\func_b_with_K_method={method}_x0={x0}_mu={mu}_2.png', bbox_inches='tight', dpi = 500)
            else:
                fig.savefig(f'{common_path}_latex\\Plot\\func_b\\colorplot\\func_b_method={method}_x0={x0}_mu={mu}_2.png', bbox_inches='tight', dpi = 500)

x0_list = [np.array([0.1,0.9])]
l0 = np.array([9.2,9.2,9.2])
z0 = np.array([0.1,0.1,0.1])
###### CONTOURPLOT ######
if True:
    for x0 in x0_list:
        for mu in mu_list:
            results = int_point(func_b, grad_b, hess_b, c_b, grad_c_b, hess_c_b, x0, method=method, alpha=1., beta=1.,
                            gamma=1., mu=mu, l0=l0, z0=z0, tol=tol, maxit=100, curv=True, seed=seed)
            k = results['n_iter']
            conv = results['convergence']
            min_point = results['x_min']
            min_value = results['f_min']
            lambda0 = tuple(np.round(results['lambda_interm'][0],1))
            z0 = tuple(np.round(results['z_interm'][0],1))
            print(f'convergence = {conv}, with {k} steps')
            print(f'starting point = {x0}, mu = {mu}')
            print(f'min point = {min_point}')
            print(f'min value = {min_value}')

            common_path = "Project_5"

            # Set size parameters for the plots
            tickparams_size = 16
            xylabel_size = 24
            suptitle_size = 35
            title_size = 19
            legend_size = 19


            delta = 0.025
            X = np.arange(-0.5, 1.5, delta)
            Y = np.arange(-0.5, 1.5, delta)
            X, Y = np.meshgrid(X,Y)
            Z = 2*X - Y**2

            fig, ax = plt.subplots(figsize=(14,9))
            
            # Obtain the contourplot
            cp = ax.contour(X, Y, Z, levels = 200)
            cb = fig.colorbar(cp, shrink=1, aspect=15, format='%.0f')
            cb.set_label(label = r'$f\,(x_{1},x_{2})$', size = 25)
            cb.ax.tick_params(labelsize=14)
            
            # Draw the constraint
            x = [0, 0, 1, 0]
            y = [0, 1, 0, 0]
            wedge = patches.Wedge((0, 0), 1, 0, 90, facecolor='lightblue')
            ax.add_patch(wedge)

            # Obtain the scatterplot of the intermediate points
            interm_points = results['x_interm']
            interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
            interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
            scatter = ax.plot(interm_x, interm_y, '-o', markersize=4, color='black')
            scatter = ax.plot(interm_x[-1], interm_y[-1], '-o', markersize=4, color='red')

            ax.tick_params(labelsize = tickparams_size)
            ax.set_xlabel(r'$x_{1}$', fontsize=xylabel_size, labelpad=10)
            ax.set_ylabel(r'$x_{2}$', fontsize=xylabel_size, labelpad=10)
            ax.set_title(fr'$x_0$={tuple(x0)}, $\lambda_0$={lambda0}, $z_0$={z0}, $\mu$ = {mu}, iter = {k}', fontsize = title_size)

            # fig.savefig(f'{common_path}_latex\\Plot\\func_b\\contourplot\\func_b_method={method}_x0={x0}_mu={mu}_seed={seed}.png', bbox_inches='tight')

            # plt.show()



