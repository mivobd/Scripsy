## Housekeeping
from types import SimpleNamespace
import json
import os
import subprocess
import numpy as np
from scipy.stats import f, norm, binom
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time

subprocess.run('cls', shell=True)
timer_start = time.perf_counter()

def tabulate_results(file_name):
    np.seterr(invalid='ignore', divide='ignore')
    with open(f"results/{file_name}.json", "r") as json_file:
        stats = SimpleNamespace(**json.load(json_file))
    stats.tsquared_violations = np.array(stats.tsquared_violations)
    stats.Q_violations = np.array(stats.Q_violations)
    stats.S_pca = np.array(stats.S_pca)

    ## Similarity stats
    # A priori labelling for evaluating performance, not available for use
    matches = np.array(stats.H_faults) == stats.snapshot_fault
    # print(stats.H_faults)
    # print(stats.snapshot_fault)
    # print(matches)
    not_matches = [int(ii) for ii in ~matches.astype(bool)]

    # Quantifying performance
    Np = np.zeros(3)
    N1 = np.zeros(3)
    N2 = np.zeros(3)
    p = np.zeros(3)
    n = np.zeros(3)
    nmax = np.zeros(3)
    e = np.zeros(3)
    Ndb = np.sum(matches) # historical windows that are actually similar

    print(f"Snapshot fault: {stats.snapshot_fault}")

    for ii, statistic in enumerate([["tsquared_violations", stats.r95],["Q_violations", stats.r95]]):#, ["Q_violations", stats.r95]]):
        candidate_pool = [int(ii) for ii in getattr(stats, statistic[0]) < statistic[1]]
        Np[ii] = np.sum(getattr(stats, statistic[0]) < statistic[1]) # candidate pool size (number of data sets classified as similar)
        N1[ii] = np.sum((getattr(stats, statistic[0]) < statistic[1]) * (matches)) # candidate pool data sets that are actually similar
        N2[ii] = np.sum((getattr(stats, statistic[0]) < statistic[1]) * (not_matches)) # candidate pool data sets that are not similar (N2 = Np - N1)
        p[ii] = N1[ii] / Np[ii] * 100
        n[ii]= N1[ii] / Ndb * 100
        nmax[ii] = np.min([Np[ii]/Ndb, 1]) * 100
        e[ii] = ( p[ii] + n[ii] ) / 2

        # print(f"\n{statistic[0]} Results:")
        # print(f"Np     {Np[ii]}\nN1     {N1[ii]}\nN2     {N2[ii]}\nNdb    {Ndb}\np      {p[ii]}\nn      {n[ii]}\nnmax   {nmax[ii]}\ne      {e[ii]}")


    # stats.S_pcaBest = 0.965
    ii=2
    candidate_pool = [int(ii) for ii in getattr(stats, "S_pca") > stats.S_pcaBest]
    Np[ii] = np.sum(stats.S_pca > stats.S_pcaBest) # candidate pool size (number of data sets classified as similar)
    N1[ii] = np.sum((stats.S_pca > stats.S_pcaBest) * (matches)) # candidate pool data sets that are actually similar
    N2[ii] = np.sum((stats.S_pca > stats.S_pcaBest) * (not_matches)) # candidate pool data sets that are not similar (N2 = Np - N1)
    p[ii] = N1[ii] / Np[ii] * 100
    n[ii]= N1[ii] / Ndb * 100
    nmax[ii] = np.min([Np[ii]/Ndb, 1]) * 100
    e[ii] = ( p[ii] + n[ii] ) / 2

    # print(f"\nS_pca Results:")
    # print(f"Np     {Np[ii]}\nN1     {N1[ii]}\nN2     {N2[ii]}\nNdb    {Ndb}\np      {p[ii]}\nn      {n[ii]}\nnmax   {nmax[ii]}\ne      {e[ii]}")

    significant_digits = 3
    Np = [f"{x:.{significant_digits}g}" if isinstance(x, float) else x for x in Np]
    N1 = [f"{x:.{significant_digits}g}" if isinstance(x, float) else x for x in N1]
    N2 = [f"{x:.{significant_digits}g}" if isinstance(x, float) else x for x in N2]
    p = [f"{x:.{significant_digits}g}" if isinstance(x, float) else x for x in p]
    n = [f"{x:.{significant_digits}g}" if isinstance(x, float) else x for x in n]
    nmax = [f"{x:.{significant_digits}g}" if isinstance(x, float) else x for x in nmax]
    e = [f"{x:.{significant_digits}g}" if isinstance(x, float) else x for x in e]

    results = {
        # "Method": ["Hotelling's T-squared", "Q statistic", "PCA similarity"],
        "Np": Np,
        "N1": N1,
        "N2": N2,
        "Ndb": Ndb,
        "p": p,
        "n": n,
        "nmax": nmax,
        "e": e
    }
    print(f"{file_name} results:\n{pd.DataFrame(results, index=["Hotelling's T-squared", "Q statistic", "PCA similarity"])}\n\n")
    return None

num = SimpleNamespace()
operating_conditons = pd.read_csv('fault_details.txt', sep=',')["Fault ID"].to_numpy()
operating_conditons = np.insert(operating_conditons, 0, "N")
colours = ["#008080", "#A52A2A", "#FFD700", "#40E0D0", "#ADD8E6", "#000080", "#90EE90", "#006400", "#FFCCCB", "#8B0000", "#E6E6FA", "#4B0082", "#FFDAB9", "#D2691E", "#FFB6C1", "#C71585", "#D3D3D3", "#4A4A4A", "#000000", "#FFA500", "#8B8000"]
num.colours = len(colours)
markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "8", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]
# fig, ax = plt.subplots(2, 2)

tabulate_results("PCA(cum_explained)")
tabulate_results("PCA(OHT)")
tabulate_results("BPCA")

print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")