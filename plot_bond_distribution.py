"""
Overlay the theoretical equilibrium bond-length distribution
    P(r) ~ r^2 exp(-V/kT),  V = 1/2 kbond (r-r0)^2
on the measured histogram from the purely-thermal dumbbell run.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_sim_dumbbells_thermal import read_trajectory, equilibrium_length_stats, minimum_image

# --- locate the run and load the parameters it saved (test_sim_dumbbells_thermal.main) ---
beta = 0.1                         # selects the run directory
data_save_dir = f"data/dumbbells_thermal_beta_{beta}"
config_dir = f"{data_save_dir}/config"

trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # NumPy 2.0 renamed trapz


def _load_params():
    """Load the simulation parameters pickled by test_sim_dumbbells_thermal.main."""
    with open(f"{data_save_dir}/params.pkl", "rb") as f:
        return pickle.load(f)


params = _load_params()
kbond = params["kbond"]
kT = params["kT"]
r0 = params["r0"]
N_dumbbell = params["N_poly"]      # dumbbells are the "polymer chains" in the sim params
box = params.get("box", None)


def measured_bond_lengths(drop_transient=True):
    times, traj = read_trajectory(config_dir)
    b = traj[:, 1:2 * N_dumbbell:2, :] - traj[:, 0:2 * N_dumbbell:2, :]
    b = minimum_image(b, box)                      # nearest image under PBC (no-op if unbounded)
    L = np.linalg.norm(b, axis=-1)                 # (frames, N_dumbbell)
    if drop_transient:                             # first frame ~ t=0 is pre-equilibrium
        L = L[times >= 0.5]
    return L.ravel()


def main(data_save_dir):
    L = measured_bond_lengths()
    n = L.size

    rmax = max(L.max() * 1.05, r0 + 6 * np.sqrt(kT / kbond))
    r = np.linspace(0.0, rmax, 2000)

    # correct length pdf (with the r^2 shell Jacobian)
    P = r**2 * np.exp(-0.5 * kbond * (r - r0)**2 / kT)
    P /= trapz(P, r)

    # naive Boltzmann factor without the r^2 factor (mode at r0) -- for contrast
    G = np.exp(-0.5 * kbond * (r - r0)**2 / kT)
    G /= trapz(G, r)

    th_mean, th_mode, th_std = equilibrium_length_stats(kbond, r0, kT)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.hist(L, bins=24, range=(0, rmax), density=True,
            color="0.75", edgecolor="0.4", alpha=0.9,
            label=f"simulation (n={n})")
    ax.plot(r, P, color="C3", lw=2.4,
            label=r"theory  $P(r)\propto r^{2}e^{-V/k_BT}$")
    ax.plot(r, G, color="C0", lw=1.6, ls="--",
            label=r"Boltzmann factor $e^{-V/k_BT}$ (no $r^{2}$)")

    ax.axvline(r0, color="k", ls=":", lw=1.2)
    ax.text(r0, ax.get_ylim()[1] * 0.92, r"  $r_0$", ha="left", va="top")
    ax.axvline(L.mean(), color="C3", ls="-.", lw=1.2,
               label=fr"$\langle r\rangle_{{\rm sim}}={L.mean():.3f}$ "
                     fr"(theory {th_mean:.3f})")

    ax.set_xlabel(r"bond length $r$")
    ax.set_ylabel(r"probability density $P(r)$")
    ax.set_title(r"Equilibrium bond-length distribution "
                 fr"($k_{{\rm bond}}={kbond:g},\ k_BT={kT:g},\ r_0={r0:g}$)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, rmax)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = f"figures/bond_length_distribution.{ext}"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    print(f"sim  <r>={L.mean():.4f}  std={L.std():.4f}  (n={n})")
    print(f"theory <r>={th_mean:.4f}  mode={th_mode:.4f}  std={th_std:.4f}")


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    beta = 0.1
    data_save_dir = f"data/dumbbells_thermal_beta_{beta}"
    main(data_save_dir=data_save_dir)
