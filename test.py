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

control = 1 # boolean, control (1) or no control (0) for the snapshot
autoreggressive_noise = 1
measurement_noise = 1
folder_path = f"BPCA_H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}"
shutil.copy2(f"save_state.json", f"{folder_path}/OldSnapshot_N.json")

print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")