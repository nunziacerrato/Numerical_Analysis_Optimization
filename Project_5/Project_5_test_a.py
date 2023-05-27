import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import pandas as pd
from decimal import Decimal
import math
from Project_5 import *

func_a = lambda x : (x[0] - 4)**2 + x[1]**2
grad_a = lambda x : np.array([2*(x[0]-4), 2*x[1]])
hess_a = lambda x : np.array([[2, 0],[0, 2]])

c_a = lambda x : np.array([2 - x[0] - x[1], x[0], x[1]])
grad_c_a = lambda x : np.array([[-1,-1],[1,0],[0,1]])
hess_c_a = lambda x : np.zeros((2,3,2))

x0 = np.array([0.1,0.1])

method = 'basic'
mu = 1e-12
tol = 1e-12
seed = 1

results = int_point(func_a, grad_a, hess_a, c_a, grad_c_a, hess_c_a, x0, method=method, alpha=1., beta=1.,
                   gamma=1., mu=mu, tol=tol, maxit=100, seed=seed)
k = results['n_iter']
conv = results['convergence']
# x0 = results['x0']
min_point = results['x_min']
min_value = results['f_min']
mu = results['mu']
print(f'convergence = {conv}, with {k} steps')
print(f'starting point = {x0}, mu = {mu}')
print(f'min point = {min_point}')
print(f'min value = {min_value}')

# print(f'result={result}')

plot = False
if plot == True:
    ##################### PLOT #####################
    common_path = "Project_5"
    # Set size parameters for the plots
    tickparams_size = 16
    xylabel_size = 20
    suptitle_size = 35
    title_size = 22
    legend_size = 19

    ########### 3D plot ###########
    delta = 0.025
    X = np.arange(0., 3.0, delta)
    Y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(X,Y)
    Z = (X - 4)**2 + Y**2
    C = X + Y - 2

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(14,9))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=True, alpha=0.7)

    # Add a color bar which maps values to colors and choose the format
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.0f}')
    cb = fig.colorbar(surf, shrink=0.5, aspect= 10)
    cb.ax.tick_params(labelsize=12) 

    # Obtain the scatterplot of the intermediate points
    interm_points = results['x_interm']
    interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
    interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
    zz = func_a([interm_x, interm_y])
    scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
    scatter = ax.plot(interm_x[-1], interm_y[-1], zz[-1], '-o', markersize=4, color='red')

    ax.tick_params(labelsize = 12)
    ax.set_xlabel(r'$x_{1}$', fontsize=xylabel_size, labelpad=10)
    ax.set_ylabel(r'$x_{2}$', fontsize=xylabel_size, labelpad=10)
    ax.set_zlabel(r'$f\,(x_{1},x_{2})$', fontsize=xylabel_size)

    # fig.savefig(f'{common_path}_latex\\Plot\\func_a_method={method}_x0={x0}_mu={mu}_seed={seed}', bbox_inches='tight')

    ########### Contour plot ###########
    delta = 0.0025
    Xc = np.arange(0, 3, delta)
    Yc = np.arange(-2, 2, delta)
    Xc, Yc = np.meshgrid(Xc,Yc)
    Zc = (Xc - 4)**2 + Yc**2
    C = Xc + Yc - 2

    fig_contour, ax_contour = plt.subplots(figsize=(14,9))
    cp = ax_contour.contour(Xc, Yc, Zc, levels = 200)
    cb = fig_contour.colorbar(cp, shrink=1, aspect=15, format='%.0f')
    cb.set_label(label = r'$f\,(x_{1},x_{2})$', size = 25)
    cb.ax.tick_params(labelsize=14) 
    scatter_c = ax_contour.plot(interm_x, interm_y, '-o', markersize=4, color='black')
    scatter_c = ax_contour.plot(interm_x[-1], interm_y[-1], '-o', markersize=4, color='red')

    ax_contour.tick_params(labelsize = tickparams_size)
    ax_contour.set_xlabel(r'$x_{1}$', fontsize=25)
    ax_contour.set_ylabel(r'$x_{2}$', fontsize=25)

    # fig_contour.savefig(f'{common_path}_latex\\Plot\\func_a_method={method}_x0={x0}_mu={mu}_seed={seed}', bbox_inches='tight')

    plt.show()
