##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

##### generate points
points = np.array([[0, 1.5, 1.8, 2.3, 3.8, 5.1],
                   [0, 0.6, 0.9, 2.4, 2.9, 4.1],
                   [0, 1.1, 2.3, 3.1, 3.6, 5.2]])

##### degree of polinomial
p = 3

#####
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(xs=points[0,:],ys=points[1,:],zs=points[2,:])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 5.01, 0.25)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2+yy**2)
f = interpolate.interp2d(x, y, z, kind='cubic')
xnew = np.arange(-5.01, 5.01, 1e-2)
ynew = np.arange(-5.01, 5.01, 1e-2)
znew = f(xnew, ynew)
plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
plt.show()