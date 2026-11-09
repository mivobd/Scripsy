import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import json
from types import SimpleNamespace
import subprocess

subprocess.run('cls', shell=True)

snapshot = "N" # string, set the operating condition of the snapshot.   "N" for NOC and peruse fault_details.txt for the fault IDs. Classifying faults did not end up being part of the research, so this will never change.
control = 1 # integer boolean, control (1) or no control (0) in the snapshot. Choice reflected in digit following "C" in dataset filename.
autoregressive_noise = 1 # integer boolean, as above, but with autoregressive noise. Choice reflected in digit following "AR" in dataset filename.
measurement_noise = 1 # integer boolean, I think you get the point. Choice reflected in digit following "M" in dataset filename.


data = SimpleNamespace()
data.snapshot = SimpleNamespace() # dedicated to snapshot-related values
stats = SimpleNamespace()
simulation_path = f"simulation_data/C{control}_AR{autoregressive_noise}_M{measurement_noise}"

# Define numerator and denominator polynomial coefficients
num = [1/16]  # e.g., 1
den = [2/60, 1]  # e.g., s^2 + 2s + 5

# Create a transfer function using SciPy signal
system = signal.TransferFunction(num, den)

# Create an array of x data (e.g., angular frequencies or time/frequency points)
with open(f"{simulation_path}/OldSnapshot_{snapshot}.json", 'r', encoding='utf-8') as file:
    data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)

x_vals = np.array(data.results["Q_CC"])
t_vals = np.array(data.results["t"])

# Evaluate frequency response: w gives frequencies, h gives complex frequency response
t_out, y_out, x_out = signal.lsim(system, U=x_vals, T=t_vals)

# Plot the magnitude response
plt.figure(figsize=(8, 5))
plt.plot(t_out, y_out, label="Magnitude |H(jwx)|", color="blue", linewidth=2)
plt.title("Transfer Function Magnitude Response")
plt.xlabel("X Data (Frequency)")
plt.ylabel("|H(x)|")
plt.grid(True)
plt.legend()
plt.show()