import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

data = np.genfromtxt('data.txt')

airgaps = np.linspace(420, 700, 501)
wavelengths = np.linspace(430, 650, 501)
energies = 1239.84/wavelengths

fig, ax = plt.subplots(1,1, figsize=(8, 6), facecolor='w', edgecolor='k', dpi=150)
ax.pcolormesh(airgaps, energies, data, cmap = 'inferno', alpha=1, shading ='gouraud', vmin=0)
ax.set_ylabel('Energy (eV)',fontsize=20)
ax.set_xlabel('Cavity length (nm)', fontsize=20)
ax.tick_params(axis='both', direction = "in", which='major', labelsize = 20,
right = True, top = True, left = True)

ax_in = ax.inset_axes([600, 2.5, 90, 0.32], transform=ax.transData)
ax_in.pcolormesh(airgaps, energies, data, cmap = 'inferno', alpha=1, shading ='gouraud', vmin=0)
ax_in.set_xlim((500, 530))
ax_in.set_ylim((2.385, 2.41))
ax_in.set_ylabel('Energy (eV)',fontsize=15)
ax_in.set_xlabel('Cavity length (nm)', fontsize=15)
ax_in.tick_params(axis='both', direction = "in", which='major', labelsize = 13,
right = True, top = True, left = True)
ax.indicate_inset_zoom(ax_in)
plt.show()
fig.savefig('FigureWithInset.png', bbox_inches='tight')
plt.cla()
plt.clf()
plt.close('all')   
