# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from cycler import cycler

monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '-.',(0, (3, 5, 1, 5, 1, 5))]) )

plt.rcParams['axes.prop_cycle'] = monochrome
plt.rcParams['font.size'] = 10


