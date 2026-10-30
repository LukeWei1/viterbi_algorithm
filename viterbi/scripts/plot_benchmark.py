#!/usr/bin/env python3
"""Turn results/bench_*.csv into PNG plots (requires matplotlib)."""
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# 1. Viterbi vs brute force ---------------------------------------------------
rows = read("results/bench_vs_brute.csv")
T = [int(r["T"]) for r in rows]
tv = [float(r["viterbi_s"]) for r in rows]
tb = [float(r["brute_s"]) for r in rows]
N = 3

fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.semilogy(T, tb, "o-", label=f"brute force  O(T·N^T)")
ax.semilogy(T, tv, "s-", label=f"Viterbi  O(T·N²)")
ax.set_xlabel("sequence length T")
ax.set_ylabel("time per decode (seconds, log scale)")
ax.set_title(f"Viterbi vs. brute force (N = {N} states)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/bench_vs_brute.png", dpi=150)

# 2. Scaling in T -------------------------------------------------------------
rows = read("results/bench_scale_T.csv")
T2 = [int(r["T"]) for r in rows]
t2 = [float(r["viterbi_s"]) for r in rows]
fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.loglog(T2, t2, "o-", label="measured")
ref = [t2[0] * (x / T2[0]) for x in T2]
ax.loglog(T2, ref, "--", color="gray", label="ideal linear in T")
ax.set_xlabel("sequence length T")
ax.set_ylabel("time per decode (seconds)")
ax.set_title("Viterbi runtime vs. T (N = 5)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/bench_scale_T.png", dpi=150)

# 3. Scaling in N -------------------------------------------------------------
rows = read("results/bench_scale_N.csv")
Ns = [int(r["N"]) for r in rows]
t3 = [float(r["viterbi_s"]) for r in rows]
fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.loglog(Ns, t3, "o-", label="measured")
k = len(Ns) // 2
ref = [t3[k] * (x / Ns[k]) ** 2 for x in Ns]
ax.loglog(Ns, ref, "--", color="gray", label="ideal quadratic in N")
ax.set_xlabel("number of states N")
ax.set_ylabel("time per decode (seconds)")
ax.set_title("Viterbi runtime vs. N (T = 500)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/bench_scale_N.png", dpi=150)

print("wrote results/bench_vs_brute.png, bench_scale_T.png, bench_scale_N.png")
