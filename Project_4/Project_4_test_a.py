''' This script implements the unconstrained optimization of the test function (a)
    using the Newton algorithms reported in the library Project_4.py.
'''
from Project_4 import *
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import seaborn as sns

#  Define the test function (a) and its gradient and hessian as lambda functions.
func_a = lambda x: (x[0]-2)**4 + ((x[0]-2)**2)*x[1]**2 + (x[1]+1)**2
grad_a = lambda x: np.array([4*(x[0]-2)**3 + 2*(x[0]-2)*(x[1]**2), 2*((x[0]-2)**2)*x[1] + 2*(x[1]+1)])
hess_a = lambda x: np.array([[12*(x[0]-2)**2 + 2*x[1]**2, 4*(x[0]-2)*x[1]],
                                  [4*(x[0]-2)*x[1], 2*(x[0]-2)**2 + 2]])
x0_a_1 = np.array([1,1])
x0_a_2 = np.array([2,-1])
sol_x_a = (2,-1)
sol_f_a = 0


# Compute the minimum value of the function using the standard Newton algorithm
results = Newton(func_a, grad_a, hess_a, 1e-5, 100, x0_a_1, sol_x_a, sol_f_a, 1, backtracking=False)

# results = Newton_trust_region(func_a, grad_a, hess_a, 1e-5, 100, x0_a_1, sol_x_a, sol_f_a, alpha=1, eta=0.01)

##################### PLOT #####################
# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 12
suptitle_size = 35
title_size = 22
legend_size = 19

########### 3D plot ###########
delta = 0.025
X = np.arange(0., 3.0, delta)
Y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(X,Y)
Z = (X-2)**4 + ((X-2)**2)*(Y**2) + (Y+1)**2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                linewidth=0, antialiased=True, alpha=0.7)

# Add a color bar which maps values to colors and choose the format
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.0f}')
fig.colorbar(surf, shrink=0.5, aspect= 10)

# Obtain the scatterplot of the intermediate points
interm_points = results['interm_point']
interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
zz = func_a([interm_x, interm_y])
scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
scatter = ax.plot(interm_x[-1], interm_y[-1], zz[-1], '-o', markersize=4, color='red')

ax.set_xlabel(r'$x_{1}$', fontsize=xylabel_size)
ax.set_ylabel(r'$x_{2}$', fontsize=xylabel_size)
ax.set_zlabel(r'$f\,(x_{1},x_{2})$', fontsize=xylabel_size)

########### Contour plot ###########
delta = 0.0025
Xc = np.arange(0, 3, delta)
Yc = np.arange(-2, 2, delta)
Xc, Yc = np.meshgrid(Xc,Yc)
Zc = (Xc-2)**4 + ((Xc-2)**2)*(Yc**2) + (Yc+1)**2

fig_contour, ax_contour = plt.subplots()
cp = ax_contour.contour(Xc, Yc, Zc, levels = 200)
cb = fig_contour.colorbar(cp, shrink=1, aspect=15, format='%.0f')
cb.set_label(label = r'$f\,(x_{1},x_{2})$', size = 14)
scatter_c = ax_contour.plot(interm_x, interm_y, '-o', markersize=4, color='black')
scatter_c = ax_contour.plot(interm_x[-1], interm_y[-1], '-o', markersize=4, color='red')

ax_contour.set_xlabel(r'$x_{1}$', fontsize=14)
ax_contour.set_ylabel(r'$x_{2}$', fontsize=14)

plt.show()