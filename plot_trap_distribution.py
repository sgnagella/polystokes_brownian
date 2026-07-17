"""
Overlay the theoretical equilibrium harmonic-trap distribution on the measured
displacement statistics of the trapped colloid from the purely-thermal run.

Trap potential  V = 1/2 ktrap |x - x_trap|^2  gives, at equilibrium:
  - each Cartesian displacement component  ~  N(0, sigma^2),  sigma^2 = kT/ktrap
  - the radial displacement r = |x - x_trap|  ~  P(r) ~ r^2 exp(-ktrap r^2 / 2 kT)
    (the r^2 is the 3D spherical-shell Jacobian; this is the r0 -> 0 analogue of
    the bond-length distribution in plot_bond_distribution.py).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_sim_dumbbells_thermal import (
    read_trajectory, equilibrium_trap_stats, colloid_displacements,
)

# --- locate the run and load the parameters it saved (test_sim_dumbbells_thermal.main) ---
fene = False
beta = 0.01                         # selects the run directory
data_save_dir = f"data/test_dumbbells_thermal_beta_{beta}_fene_{fene}"
config_dir = f"{data_save_dir}/config"
drop_transient = 0.0               # discard this leading fraction (relax from r=0 start)

trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # NumPy 2.0 renamed trapz


def _load_params():
    """Load the simulation parameters pickled by test_sim_dumbbells_thermal.main."""
    with open(f"{data_save_dir}/params.pkl", "rb") as f:
        return pickle.load(f)


params = _load_params()
kT = params["kT"]
ktrap = params["ktrap"]
N_trapped = params["N_trapped"]    # trailing particles that are trapped colloids
box = params.get("box", None)


def measured_displacements():
    """Colloid displacement components and radial displacement (transient dropped)."""
    times, traj = read_trajectory(config_dir)
    disp, _ = colloid_displacements(times, traj, N_trapped, box, drop_transient)
    comps = disp.reshape(-1, 3).ravel()             # pooled Cartesian components
    r = np.linalg.norm(disp, axis=-1).ravel()       # radial displacements
    return comps, r


def main():
    comps, r = measured_displacements()
    sigma, r_mean, r_mode, r_std = equilibrium_trap_stats(ktrap, kT)

    fig, (axc, axr) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- per-component Gaussian ----
    xhi = max(abs(comps).max() * 1.05, 4.0 * sigma)
    xg = np.linspace(-xhi, xhi, 400)
    gauss = np.exp(-xg**2 / (2.0 * sigma**2)) / np.sqrt(2.0 * np.pi * sigma**2)
    axc.hist(comps, bins=40, range=(-xhi, xhi), density=True,
             color="0.75", edgecolor="0.4", alpha=0.9,
             label=f"simulation (n={comps.size})")
    axc.plot(xg, gauss, color="C3", lw=2.4,
             label=r"theory  $\mathcal{N}(0,\,k_BT/k_{\rm trap})$")
    axc.axvline(0.0, color="k", ls=":", lw=1.2)
    axc.set_xlabel(r"displacement component $x-x_{\rm trap}$")
    axc.set_ylabel("probability density")
    axc.set_title(fr"Cartesian displacement ($\sigma={sigma:.3g}$)")
    axc.legend(frameon=False, fontsize=9)

    # ---- radial distribution ----
    rmax = max(r.max() * 1.05, r_mean + 4.0 * r_std)
    rr = np.linspace(0.0, rmax, 2000)
    P = rr**2 * np.exp(-0.5 * ktrap * rr**2 / kT)      # correct radial pdf (r^2 shell)
    P /= trapz(P, rr)
    G = np.exp(-0.5 * ktrap * rr**2 / kT)              # naive Boltzmann factor (no r^2)
    G /= trapz(G, rr)
    axr.hist(r, bins=30, range=(0, rmax), density=True,
             color="0.75", edgecolor="0.4", alpha=0.9,
             label=f"simulation (n={r.size})")
    axr.plot(rr, P, color="C3", lw=2.4,
             label=r"theory  $P(r)\propto r^{2}e^{-k_{\rm trap} r^{2}/2k_BT}$")
    axr.plot(rr, G, color="C0", lw=1.6, ls="--",
             label=r"Boltzmann factor $e^{-k_{\rm trap} r^{2}/2k_BT}$ (no $r^{2}$)")
    axr.axvline(r.mean(), color="C3", ls="-.", lw=1.2,
                label=fr"$\langle r\rangle_{{\rm sim}}={r.mean():.3g}$ (theory {r_mean:.3g})")
    axr.set_xlabel(r"radial displacement $r=|x-x_{\rm trap}|$")
    axr.set_ylabel(r"probability density $P(r)$")
    axr.set_title(fr"Radial displacement ($k_{{\rm trap}}={ktrap:g},\ k_BT={kT:g}$)")
    axr.legend(frameon=False, fontsize=9)
    axr.set_xlim(0, rmax)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = f"figures/trap_distribution.{ext}"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    print(f"sim  sigma={comps.std():.4f}  <r>={r.mean():.4f}  std={r.std():.4f}  "
          f"(n={r.size})")
    print(f"theory sigma={sigma:.4f}  <r>={r_mean:.4f}  mode={r_mode:.4f}  std={r_std:.4f}")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    main()
