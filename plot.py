import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

x = np.array([0.25, 0.27, 0.30, 0.32, 0.35])
y = np.array([-11, -4, -1, 6, 13])

xnew = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(xnew)

plt.plot(xnew, y_smooth)
plt.show()