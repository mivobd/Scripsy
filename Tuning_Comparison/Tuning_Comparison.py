## Housekeeping
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
subprocess.run('cls', shell=True)

"""
This file generates the plots to compare the control performance when using the original versus retuned controller parameters.
"""


def plot_results(results, fig, ax):
    num_recorded_points = len(results["t"])
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = ax

    # level
    ax1.plot(results["t"], results["h"], label="measurement", color="blue")
    ax1.plot(results["t"], 6*np.ones(num_recorded_points), label="setpoint", color="orange")
    ax1.set_ylim([5, 7])
    ax1.set_ylabel("Level [m]")
    ax1.set_xlim([0, 120])
    ax1.set_xlabel("Time [min]")

    # outlet flow
    ax2.plot(results["t"], results["Q"], color="blue")
    ax2.plot(results["t"], 100+np.array(results["hC"]), color="orange")
    ax2.set_ylim([85, 115])
    ax2.set_ylabel(r"Flow rate [L min$^{-1}$]")
    ax2.set_xlim([0, 120])
    ax2.set_xlabel("Time [min]")

    # outlet valve
    ax3.plot(results["t"], np.array(results["l"])*100, color="blue")
    ax3.set_ylim([0, 100])
    ax3.set_ylabel("Opening [%]")
    ax3.set_xlim([0, 120])
    ax3.set_xlabel("Time [min]")

    # temperature
    ax4.plot(results["t"], results["T"], color="blue")
    ax4.plot(results["t"], 402.35*np.ones(num_recorded_points), color="orange")
    ax4.set_ylim([382.35, 422.35])
    ax4.set_ylabel("Temperature [K]")
    ax4.set_xlim([0, 120])
    ax4.set_xlabel("Time [min]")

    # coolant flow
    ax5.plot(results["t"], results["Q_C"], color="blue")
    ax5.plot(results["t"], 15+np.array(results["TC"]), color="orange")
    ax5.set_ylim([0, 30])
    ax5.set_ylabel(r"Flow rate [L min$^{-1}$]")
    ax5.set_xlim([0, 120])
    ax5.set_xlabel("Time [min]")

    # coolant valve
    ax6.plot(results["t"], np.array(results["l_C"])*100, color="blue")
    ax6.set_ylim([0, 100])
    ax6.set_ylabel("Opening [%]")
    ax6.set_xlim([0, 120])
    ax6.set_xlabel("Time [min]")

    fig.legend(loc='outside upper center', ncols=2)

    return None


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


plt.show()