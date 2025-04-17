import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = x ** x
plt.plot(x, y, 'r-', mec = 'k')
plt.show()