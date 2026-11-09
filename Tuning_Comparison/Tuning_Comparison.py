## Housekeeping
import json
import subprocess
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

subprocess.run('cls', shell=True)

"""
This file generates the plots to compare the control performance when using the original versus retuned controller parameters.
"""


def plot_results(results, fig, ax):
    num_recorded_points = len(results["t"])
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = ax

    # to get valve level action
    system = signal.TransferFunction([1/16], [2/60, 1])

    # level
    ax1.plot(results["t"], results["h"], label="measurement", color="blue")
    ax1.plot(results["t"], 6 * np.ones(num_recorded_points), label="setpoint", color="orange")
    ax1.set_ylim([5, 7])
    ax1.set_ylabel("Level [m]")
    ax1.set_xlim([0, results["t"][-1]])
    ax1.set_xlabel("Time [min]")

    # outlet flow
    ax2.plot(results["t"], results["Q"], color="blue")
    ax2.plot(results["t"], 100 + np.array(results["hC"]), color="orange")
    ax2.set_ylim([85, 115])
    ax2.set_ylabel(r"Flow rate [L min$^{-1}$]")
    ax2.set_xlim([0, results["t"][-1]])
    ax2.set_xlabel("Time [min]")

    # outlet valve
    t_out, l, x_out = signal.lsim(system, U=results["QC"], T=results["t"])
    ax3.plot(results["t"], l * 100 + 50, color="blue")
    ax3.set_ylim([0, 100])
    ax3.set_ylabel("Opening [%]")
    ax3.set_xlim([0, results["t"][-1]])
    ax3.set_xlabel("Time [min]")

    # temperature
    ax4.plot(results["t"], results["T"], color="blue")
    ax4.plot(results["t"], 402.35 * np.ones(num_recorded_points), color="orange")
    ax4.set_ylim([382.35, 422.35])
    ax4.set_ylabel("Temperature [K]")
    ax4.set_xlim([0, results["t"][-1]])
    ax4.set_xlabel("Time [min]")

    # coolant flow
    ax5.plot(results["t"], results["Q_C"], color="blue")
    ax5.plot(results["t"], 15 + np.array(results["TC"]), color="orange")
    ax5.set_ylim([0, 30])
    ax5.set_ylabel(r"Flow rate [L min$^{-1}$]")
    ax5.set_xlim([0, results["t"][-1]])
    ax5.set_xlabel("Time [min]")

    # coolant valve
    t_out, l_C, x_out = signal.lsim(system, U=results["Q_CC"], T=results["t"])
    ax6.plot(results["t"], l_C * 100 + 50, color="blue")
    ax6.set_ylim([0, 100])
    ax6.set_ylabel("Opening [%]")
    ax6.set_xlim([0, results["t"][-1]])
    ax6.set_xlabel("Time [min]")


# Original parameters from Wong (2006)
with open(f"Tuning_Comparison/Original.json", "r") as json_file:
    Original_data = json.load(json_file)

fig1, ax_Original = plt.subplots(2, 3, figsize=(12, 8), layout='constrained', num='Original Controller Parameters')
plot_results(Original_data, fig1, ax_Original)

# Retuned parameters
with open(f"Tuning_Comparison/Retuned.json", "r") as json_file:
    Retuned_data = json.load(json_file)

fig2, ax_Retuned = plt.subplots(2, 3, figsize=(12, 8), layout='constrained', num='Retuned Controller Parameters')
plot_results(Retuned_data, fig2, ax_Retuned)

# Retuned parameters
with open(f"simulation_data/C0_AR0_M1/run_1.json", "r") as json_file:
    ss_data = json.load(json_file)

fig3, ax_ss = plt.subplots(2, 3, figsize=(12, 8), layout='constrained', num='Steady State with Measurement Noise')
plot_results(ss_data, fig3, ax_ss)

plt.show()