#include "arrays.h"
#include "data.h"
#include "params.h"
#include "pair_interaction.h"
#include <iostream>
#include <cmath>

using namespace arrays;

void pair_interaction(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct){
    // Computes pair interactions using the Verlet list.
    //
    // Weeks-Chandler-Andersen (WCA) potential: a Lennard-Jones potential truncated
    // and shifted at its minimum r_cut = 2^(1/6) * sigma, giving a purely repulsive
    // excluded-volume interaction. Applied to
    //   - colloid-monomer pairs (type_id == 1)
    //   - colloid-colloid  pairs (type_id == 2)
    // Monomer-monomer pairs (type_id == 0) are skipped.
    //
    //   U(r) = 4*eps*[ (sigma/r)^12 - (sigma/r)^6 ] + eps,  r < r_cut
    //   F(r) = -dU/dr = (24*eps/r) * [ 2*(sigma/r)^12 - (sigma/r)^6 ]   (repulsive)
    //
    // Forces are accumulated into fext with the same sign convention as the trapping
    // and bond forces (fext stores the negative of the physical force).

    int kk, jj, kk3, jj3, type_id;
    PetscInt& ndimp = consts.ndimp;
    PetscInt& Np = pinfo.Np;
    double& epsilon = pinfo.epsilon;
    double dr, dr_inv, rcut, sig, sig_dr, sig2, sig6, sig12, fmag, fx, fy, fz;
    std::vector<float> &rcuts = dataStruct.rcuts;
    std::vector<float> &sigmas = dataStruct.sigmas;

    for( kk = 0; kk < (Np-1); kk++ ){
        for( int pidx : vlist[kk] ){
            type_id = pair_types[pidx];

            // Skip monomer-monomer pairs (no excluded-volume WCA here).
            if (type_id == 0){
                continue;
            }

            jj = id[1][pidx];

            sig    = sigmas[type_id];
            rcut   = rcuts[type_id];
            dr     = pd[3][pidx];
            dr_inv = pd[4][pidx];

            // WCA vanishes beyond the cutoff r_cut = 2^(1/6) * sigma.
            if( dr >= rcut ){
                continue;
            }

            // Repulsive WCA force magnitude F(r) = (24*eps/r)*(2*(s/r)^12 - (s/r)^6).
            sig_dr = sig * dr_inv;
            sig2   = sig_dr * sig_dr;
            sig6   = sig2 * sig2 * sig2;
            sig12  = sig6 * sig6;
            fmag   = 24.0 * epsilon * dr_inv * ( 2.0 * sig12 - sig6 );

            // Separation unit vector points from jj to kk (dx = x_kk - x_jj), so a
            // positive fmag pushes kk away from jj (repulsion).
            fx = fmag * pd[0][pidx];
            fy = fmag * pd[1][pidx];
            fz = fmag * pd[2][pidx];

            kk3 = ndimp * kk;
            jj3 = ndimp * jj;

            // fext stores the negative of the physical force (cf. trapping_forces).
            fext[kk3]   -= fx;
            fext[kk3+1] -= fy;
            fext[kk3+2] -= fz;
            fext[jj3]   += fx;
            fext[jj3+1] += fy;
            fext[jj3+2] += fz;
        }
    }
}
