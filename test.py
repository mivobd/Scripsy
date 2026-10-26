## Housekeeping
from types import SimpleNamespace
import json
import subprocess
import os
import numpy as np
from scipy.stats import f, norm, binom
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time
import shutil
from pathlib import Path

subprocess.run('cls', shell=True)
timer_start = time.perf_counter()

test = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
I_test = np.identity(3)
print(np.linalg.inv(test))
# print(la.cho_solve(la.cho_factor(test, lower=True), I_test))
print(np.linalg.solve(test, I_test))

print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")