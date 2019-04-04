import numpy as np
import matplotlib.pyplot as plt

size = 100
sigma_x = 1.
sigma_y = 1.

x = np.linspace(-6, 6, size)
y = np.linspace(-6, 6, size)

x, y = np.meshgrid(x, y)
z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
     + y**2/(2*sigma_y**2))))
z1 = z * -2 * (1 - (x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)))

x0 = 0
y = np.linspace(-6, 6, size)
zt = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x0**2/(2*sigma_x**2)
     + y**2/(2*sigma_y**2))))
z1t = zt *  -2 * (1 - (x0**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)))

ax0 = plt.subplot(221)
ax0.imshow(z)
ax1 = plt.subplot(222)
ax1.plot(y,zt)
ax2 = plt.subplot(223)
ax2.imshow(z1)
ax3 = plt.subplot(224)
ax3.plot(y,z1t)
plt.savefig("Log.png")
plt.show()

