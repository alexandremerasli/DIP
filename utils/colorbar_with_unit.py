import numpy as np
import matplotlib.pylab as plt 

dat = np.random.randn(10,10)
plt.imshow(dat, interpolation='none',vmax=17000 ,vmin=0,cmap="gray_r")

clb = plt.colorbar()
clb.ax.set_title('(Bq.mm$^{-3}$)')
plt.axis('off')
plt.savefig('colorbar_with_unit.png')
# plt.show()