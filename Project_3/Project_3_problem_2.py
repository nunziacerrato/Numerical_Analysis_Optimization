'''
'''

import numpy as np
import matplotlib.pyplot as plt

# Compute the eivenvalues of T_N
N = 21
index = np.array(range(1,N+1))
eigvals = 2 * (1 - np.cos((np.pi*index)/(N+1)) )

# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 18
suptitle_size = 35
title_size = 22
legend_size = 19

# Plot the eigenvalues of T_N
fig_eigvals, ax_eigvals = plt.subplots()
ax_eigvals.scatter(index, eigvals)
ax_eigvals.set_title(fr'Eigenvalues of $T_N$ with N={N}', fontsize=title_size)
ax_eigvals.tick_params(labelsize = tickparams_size)
ax_eigvals.set_xlabel('j',fontsize=20)
ax_eigvals.set_ylabel(r'$\mu_j$',fontsize=20)

index_list = [1,2,3,5,11,21]
eigenvectors_list = [ np.sqrt(2/(N+1)) * np.sin(j*index*np.pi/(N+1))  for j in index_list]

# Plot the eigenvectors of T_N
fig_eigvals, ax_eigvects = plt.subplots(nrows=3, ncols=2, constrained_layout = True)
fig_eigvals.suptitle(fr'Eigenvectors of $T_N$ with N={N}', fontsize=title_size)

ax_eigvects[0,0].scatter(index, eigenvectors_list[0])
ax_eigvects[0,0].plot(index, eigenvectors_list[0], linestyle='dotted')
# ax_eigvects[0,0].set_title(fr'Eigenvector of $T_N$ with j={index_list[0]}')
ax_eigvects[0,0].set_xlabel('k', fontsize=xylabel_size)
ax_eigvects[0,0].set_ylabel(fr'$u_{index_list[0]}$(k)', fontsize=xylabel_size)

ax_eigvects[0,1].scatter(index, eigenvectors_list[1])
ax_eigvects[0,1].plot(index, eigenvectors_list[1], linestyle='dotted')
# ax_eigvects[0,1].set_title(fr'Eigenvector of $T_N$ with j={index_list[1]}')
ax_eigvects[0,1].set_xlabel('k', fontsize=xylabel_size)
ax_eigvects[0,1].set_ylabel(fr'$u_{index_list[1]}$(k)', fontsize=xylabel_size)

ax_eigvects[1,0].scatter(index, eigenvectors_list[2])
ax_eigvects[1,0].plot(index, eigenvectors_list[2], linestyle='dotted')
# ax_eigvects[1,0].set_title(fr'Eigenvector of $T_N$ with j={index_list[2]}')
ax_eigvects[1,0].set_xlabel('k', fontsize=xylabel_size)
ax_eigvects[1,0].set_ylabel(fr'$u_{index_list[2]}$(k)', fontsize=xylabel_size)

ax_eigvects[1,1].scatter(index, eigenvectors_list[3])
ax_eigvects[1,1].plot(index, eigenvectors_list[3], linestyle='dotted')
# ax_eigvects[1,1].set_title(fr'Eigenvector of $T_N$ with j={index_list[3]}')
ax_eigvects[1,1].set_xlabel('k', fontsize=xylabel_size)
ax_eigvects[1,1].set_ylabel(fr'$u_{index_list[3]}$(k)', fontsize=xylabel_size)

ax_eigvects[2,0].scatter(index, eigenvectors_list[4])
ax_eigvects[2,0].plot(index, eigenvectors_list[4],linestyle='dotted')
# ax_eigvects[2,0].set_title(fr'Eigenvector of $T_N$ with j={index_list[4]}')
ax_eigvects[2,0].set_xlabel('k', fontsize=xylabel_size)
ax_eigvects[2,0].set_ylabel(fr'$u_{{{index_list[4]}}}$(k)', fontsize=xylabel_size)

ax_eigvects[2,1].scatter(index, eigenvectors_list[5])
ax_eigvects[2,1].plot(index, eigenvectors_list[5], linestyle='dotted')
# ax_eigvects[2,1].set_title(fr'Eigenvector of $T_N$ with j={index_list[5]}')
ax_eigvects[2,1].set_xlabel('k', fontsize=xylabel_size)
ax_eigvects[2,1].set_ylabel(fr'$u_{{{index_list[5]}}}$(k)', fontsize=xylabel_size)

plt.show()
