import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import pandas as pd
from decimal import Decimal
from Project_5 import *

func_a = lambda x : (x[0] - 4)**2 + x[1]**2
grad_a = lambda x : np.array([2*(x[0]-4), 2*x[1]])
hess_a = lambda x : np.array([[2, 0],[0, 2]])

c_a = lambda x : np.array([2 - x[0] - x[1], x[0], x[1]])
grad_c_a = lambda x : np.array([[-1,-1],[1,0],[0,1]])
hess_c_a = lambda x : np.zeros((2,3,2))


tol = 1e-12
mu = 1e-12
seed = 2
lambda_old = np.random.uniform(low=1e-16, high= 10., size=3)
z_old = np.random.uniform(low=1e-16, high= 10., size=3)
x0 = np.array([0.5,0.5])
method_list = ['basic', 'first', 'full']
method_list = ['basic']
for method in method_list:
    t0 = time.time()
    results = int_point(func_a, grad_a, hess_a, c_a, grad_c_a, hess_c_a, x0, method=method, 
                        alpha=1., beta=1., gamma=1., mu=mu, tol=tol, maxit=100, 
                        l0=lambda_old, z0=z_old, seed=seed)
    elapsed_time = time.time() - t0

    k = results['n_iter']
    conv = results['convergence']
    min_point = tuple(results['x_min'])
    min_value = results['f_min']
    lambda0 = tuple(np.round(results['lambda_interm'][0],1))
    z0 = tuple(np.round(results['z_interm'][0],1))
    print(f'convergence = {conv}, with {k} steps and method = {method}')
    # with {elapsed_time:.2g} s
    print(f'starting point = {tuple(x0)}')
    print(f'mu = {mu}, lambda_0 = {lambda0}, z_0 = {z0}')
    print(f'min point = {min_point}')
    print(f'min value = {min_value}')
    lambda_fin = tuple((results['lambda_interm'][-1]))
    z_fin = tuple((results['z_interm'][-1]))
    print(f'lambda* = {lambda_fin}')
    print(f'z* = {z_fin}')
    # print('##############################################')
    # print(f'x_0 = {tuple(x0)}, lambda_0 = {lambda0}, z_0 = {z0}')
    # print(f'mu = {mu}')
    # print(f'min point = {min_point}')
    # print(f'min value = {min_value}')
    # lambda_fin = tuple((results['lambda_interm'][-1]))
    # z_fin = z0 = tuple((results['z_interm'][-1]))
    # print(f'lambda* = {lambda_fin}')
    # print(f'z* = {z_fin}')

# How many steps are required when changing mu?
mu_list = np.arange(1e-16,0.01,1e-5)
mu_list = 10.**(np.arange(-16,1,1))
fig, ax = plt.subplots(figsize=(14,9))
seed = 10
method = 'basic'
common_path = 'Project_5'
x0 = np.array([0.3,0.7])
for mu in mu_list:
    results = int_point(func_a, grad_a, hess_a, c_a, grad_c_a, hess_c_a, x0, method=method, 
                        alpha=1., beta=1., gamma=1., mu=mu, tol=tol, maxit=100, seed=seed)
    k = results['n_iter']
    ax.scatter(np.log10(mu),k, c='b')

# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 24
suptitle_size = 35
title_size = 19
legend_size = 19
ax.tick_params(labelsize = tickparams_size)
ax.set_xlabel(r'$log_{10}(\mu)$', fontsize = xylabel_size)
ax.set_ylabel('k', fontsize = xylabel_size)
lambda0 = tuple(np.round(results['lambda_interm'][0],1))
z0 = tuple(np.round(results['z_interm'][0],1))
ax.set_title(fr'Number of iterations for $x_0$={tuple(x0)}, $\lambda_0$={lambda0}, $z_0$={z0}', fontsize = title_size)
fig.savefig(f'{common_path}_latex\\Plot\\number_iter_method={method}_x0={x0}_seed={seed}.png', bbox_inches='tight')
print(f'starting point = {tuple(x0)}')
print(f'mu = {mu}, lambda_0 = {lambda0}, z_0 = {z0}')
plt.show()


plot = False
if plot == True:
    common_path = "Project_5"
    method = 'basic'
    seed = 2
    
    # Set size parameters for the plots
    tickparams_size = 16
    xylabel_size = 24
    suptitle_size = 35
    title_size = 19
    legend_size = 19

    # Set the meshgrid 
    delta = 0.025
    X = np.arange(-0.5, 2.5, delta)
    Y = np.arange(-0.5, 2.5, delta)
    X, Y = np.meshgrid(X,Y)
    Z = (X - 4)**2 + Y**2

    mu_list = [0,1e-12,1e-3]
    x0_list = [np.array([1.,1.]),np.array([0.1,0.1]),np.array([0.1,0.9]),np.array([0.9,0.1]),
               np.array([0.3,0.7]),np.array([0.7,0.3]),np.array([0.5,0.5]),np.array([0,0]),
               np.array([2,0]),np.array([0,2])]
    x0_list = [np.array([1.,1.])]

    for x0 in x0_list:
        for mu in mu_list:
            results = int_point(func_a, grad_a, hess_a, c_a, grad_c_a, hess_c_a, x0, method=method, 
                                alpha=1., beta=1., gamma=1., mu=mu, tol=tol, maxit=100, seed=seed)
            k = results['n_iter']
            lambda0 = tuple(np.round(results['lambda_interm'][0],1))
            z0 = tuple(np.round(results['z_interm'][0],1))

            fig, ax = plt.subplots(figsize=(14,9))
            
            # Obtain the contourplot
            cp = ax.contour(X, Y, Z, levels = 200)
            cb = fig.colorbar(cp, shrink=1, aspect=15, format='%.0f')
            cb.set_label(label = r'$f\,(x_{1},x_{2})$', size = 25)
            cb.ax.tick_params(labelsize=14)
            
            # Draw the constraint
            x = [0, 0, 2, 0]
            y = [0, 2, 0, 0]
            ax.plot(x, y)
            ax.fill(x, y, 'lightblue')

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

            fig.savefig(f'{common_path}_latex\\Plot\\func_a_method={method}_x0={x0}_mu={mu}_seed={seed}.png', bbox_inches='tight')

            # plt.show()
