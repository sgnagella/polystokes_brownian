#ifndef PAIR_INT_H
#define PAIR_INT_H
#include <petscmat.h>
#include "params.h"
#include "data.h"
#include <cmath>

// Repulsive WCA force MAGNITUDE F(r) = (24*eps/r) * (2*(sigma/r)^12 - (sigma/r)^6)
// for r < rcut = 2^(1/6)*sigma; returns 0 at/beyond the cutoff. Multiply by the
// separation unit vector (pointing from the partner toward the particle) to get the
// repulsive force on that particle. Shared by pair_interaction (AB/BB) and the
// monomer cell list (AA) so the WCA physics lives in one place.
inline double wca_force_mag(double dr, double dr_inv, double sig, double rcut, double eps){
    if (dr >= rcut) return 0.0;
    double sig_dr = sig * dr_inv;
    double sig2   = sig_dr * sig_dr;
    double sig6   = sig2 * sig2 * sig2;
    double sig12  = sig6 * sig6;
    return 24.0 * eps * dr_inv * ( 2.0 * sig12 - sig6 );
}

// Purely-repulsive high-exponent (n=48) Lennard-Jones force MAGNITUDE, a much steeper,
// near-hard-sphere alternative to WCA for suppressing colloid-monomer overlaps that
// otherwise drive the mobility operator non-positive-definite (O(0.1-1) negative
// eigenvalues under the sqrt). Truncated at its minimum r = sig_eff, so it is purely
// repulsive with no attractive tail.
//
//   U(r) = eps*[ (sig_eff/r)^96 - 2*(sig_eff/r)^48 ],          r < sig_eff
//   F(r) = -dU/dr = (96*eps/r) * s*(s-1),  s = (sig_eff/r)^48   (>= 0 inside the core)
//
// sig_eff = sig*(1 + HIGHEXP_REG) applies a regularization shift to the contact distance
// so particles are held clear of true contact (originally a 1% shift for a monodisperse
// suspension; 10% here). Same sign convention as wca_force_mag: positive magnitude,
// multiply by the unit vector pointing from the partner toward the particle.
inline double highexp_force_mag(double dr, double dr_inv, double sig, double eps){
    constexpr double HIGHEXP_REG = 0.1;          // regularization shift on the interaction distance
    double sig_eff = sig * (1.0 + HIGHEXP_REG);
    if (dr >= sig_eff) return 0.0;
    double x   = sig_eff * dr_inv;               // sig_eff / r  (> 1 inside the core)
    double x2  = x * x;
    double x4  = x2 * x2;
    double x8  = x4 * x4;
    double x16 = x8 * x8;
    double s   = x16 * x16 * x16;                // (sig_eff/r)^48
    return 96.0 * eps * dr_inv * s * (s - 1.0);
}

// Asymmetric-harmonic ("potential-free" / Heyes-Melrose) hard-sphere repulsion. Acts ONLY
// on overlapping pairs (r < sig_eff) with a linear restoring force F(r) = k*(sig_eff - r);
// zero otherwise. The spring constant is calibrated as k = 1/dt so the resulting explicit
// Euler displacement removes ~the current overlap in a single step, INDEPENDENT of dt --
// unlike a stiff LJ, whose force diverges near contact and overshoots at fixed dt. sig_eff
// = sig*(1 + HS_REG) shifts the contact distance out slightly (1%) for numerical
// stability. Same sign convention as wca_force_mag: positive magnitude, multiply by the
// unit vector pointing from the partner toward the particle.
//
// (A near-field-lubrication regime would need k ~ O(1e3)/dt to overcome the divergent drag;
// this build is far-field RPY only, so k = 1/dt.)
// k is the spring constant (see pair_interaction_hs for its calibration, e.g. k = 1/dt).
inline double hs_force_mag(double dr, double sig, double k){
    constexpr double HS_REG = 0.01;         // 1% contact-distance shift
    double sig_eff = sig * (1.0 + HS_REG);
    if (dr >= sig_eff) return 0.0;
    return k * (sig_eff - dr);               // > 0 for overlap => repulsive
}

inline double hs_exp_force_mag(double dr, double sig, double k){
    constexpr double HS_REG = 0.01;         // 1% contact-distance shift
    double sig_eff = sig * (1.0 + HS_REG);
    double exp = std::exp(-k * (dr - sig_eff));
    return exp/(1-exp);
}

void pair_interaction(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev);
void pair_interaction_highexp(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev);
void pair_interaction_hs(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev, double dt);
void pair_interaction_hs_exp(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev);
#endif