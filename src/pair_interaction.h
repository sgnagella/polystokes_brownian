#ifndef PAIR_INT_H
#define PAIR_INT_H
#include <petscmat.h>
#include "params.h"
#include "data.h"

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

void pair_interaction(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev);
#endif