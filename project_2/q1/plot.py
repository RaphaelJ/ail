import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.stats import norm

step = 0.001
xs = np.arange(0, 2, step)

sigma = 0.1

rs = [0.7, 0.5]
colors = ['b', 'g']
labels = ["y = 0", "y = 1"]

fig, ax1 = plt.subplots()

def f(x, r):
    return (norm.pdf(x, r, sigma) + norm.pdf(x, 2 * r, sigma)) / 2

for i in range(0, 2):
    r = rs[i]
    color = colors[i]

    ys = f(xs, r)
    ax1.plot(xs, ys, color=color, linewidth=1, label=labels[i])

    other_r = rs[(i + 1) % 2]

    def background_f(x):
        if f(x, r) >= f(x, other_r):
            return 2
        else:
            return 0
    background = np.vectorize(background_f)(xs)

    ax1.fill_between(xs, 0, background, alpha=0.2, color=color, linewidth=0)

ax1.set_ylabel("Probability density")
ax1.set_xlabel("Radius")
ax1.legend()

def error_f(x):
    return min(f(x, rs[0]), f(x, rs[1]))

error = np.empty(xs.size)
cum_integral = 0
for i, x in enumerate(xs):
    cum_integral += error_f(x) * step
    error[i] = cum_integral

ax2 = ax1.twinx()
ax2.plot(xs, error, color='r', linewidth=1)

ax2.set_ylabel("Cumulative error")

plt.legend()

print("Integrale:", integrate.quad(error_f, -100, 100))

plt.show()
