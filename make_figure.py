"""Compare the five controllers. Produces controller_comparison.png."""

import numpy as np
import matplotlib.pyplot as plt
from soc_control import run, CONTROLLERS

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})

COLORS = {"baseline": "#888888", "I_inversion": "#4C72B0", "II_thermostat": "#DD8452",
          "III_distance": "#55A868", "IV_predictive": "#C44E52", "V_ode": "#8172B3"}
STYLES = {"baseline": ":", "I_inversion": "-", "II_thermostat": "-",
          "III_distance": "-", "IV_predictive": "--", "V_ode": "-."}

STEPS, ETA0 = 2500, 0.05

runs = {"baseline": run(None, steps=STEPS, eta0=ETA0)}
for name in CONTROLLERS:
    runs[name] = run(name, steps=STEPS, eta0=ETA0)

# Same two methods with the eta*S > 1 gate removed -> silent failure
ungated = {n: run(n, steps=STEPS, eta0=ETA0, gate=False)
           for n in ("IV_predictive", "V_ode")}

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

# (a) eta*S / 2  -- 1.0 is the critical surface
for n, r in runs.items():
    ax[0, 0].plot(r["t"], r["etaS"] / 2, color=COLORS[n], ls=STYLES[n], lw=1.4, label=n)
ax[0, 0].axhline(1.0, color="k", ls="--", lw=0.8)
ax[0, 0].set(xlabel="step $t$", ylabel=r"$\eta_t S_t/2$",
             title="a  All five controllers reach the critical surface")
ax[0, 0].legend(fontsize=7, ncol=2)

# (b) running mean of lambda
for n, r in runs.items():
    ax[0, 1].plot(r["t"], np.cumsum(r["lam"]) / np.arange(1, len(r["lam"]) + 1),
                  color=COLORS[n], ls=STYLES[n], lw=1.4)
ax[0, 1].axhline(0.0, color="k", ls="--", lw=0.8)
ax[0, 1].set(xlabel="step $t$", ylabel=r"running $\langle\lambda_t\rangle$",
             title=r"b  Stability exponent driven to $\lambda^*=0$")

# (c) learning rate
for n, r in runs.items():
    ax[1, 0].plot(r["t"], r["eta"], color=COLORS[n], ls=STYLES[n], lw=1.4)
ax[1, 0].set(xlabel="step $t$", ylabel=r"$\eta_t$", yscale="log",
             title="c  Learning-rate schedules produced")

# (d) the silent failure
for n, r in ungated.items():
    ax[1, 1].plot(r["t"], r["etaS"] / 2, color=COLORS[n], ls="-", lw=1.6,
                  label=f"{n} (no gate)")
    ax[1, 1].plot(runs[n]["t"], runs[n]["etaS"] / 2, color=COLORS[n], ls=":",
                  lw=1.2, label=f"{n} (gated)")
ax[1, 1].axhline(1.0, color="k", ls="--", lw=0.8)
ax[1, 1].axhline(0.0, color="r", ls="--", lw=0.8)
ax[1, 1].set(xlabel="step $t$", ylabel=r"$\eta_t S_t/2$",
             title=r"d  Spurious root: $\lambda=0$ also holds at $\eta S=0$")
ax[1, 1].legend(fontsize=7)

for a in ax.ravel():
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/controller_comparison.png", bbox_inches="tight")

print(f"{'method':16s} {'lam_top':>9s} {'eta*S':>8s} {'eta':>8s} {'test err':>9s}")
for n, r in runs.items():
    print(f"{n:16s} {r['lam_top']:+9.4f} {r['etaS'][-1]:8.4f} "
          f"{r['eta'][-1]:8.4f} {r['test_error']:9.4f}")
for n, r in ungated.items():
    print(f"{n+' NOGATE':16s} {r['lam_top']:+9.4f} {r['etaS'][-1]:8.4f} "
          f"{r['eta'][-1]:8.4f} {r['test_error']:9.4f}   <-- silent failure")
