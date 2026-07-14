#include "arrays.h"
#include "data.h"
#include "params.h"
#include "pair_interaction.h"
#include <iostream>
#include <cmath>

using namespace arrays;

void pair_interaction(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev){
    // Computes pair interactions using the Verlet list.
    //
    // Weeks-Chandler-Andersen (WCA) potential: a Lennard-Jones potential truncated
    // and shifted at its minimum r_cut = 2^(1/6) * sigma, giving a purely repulsive
    // excluded-volume interaction. Applied to
    //   - colloid-monomer pairs (type_id == 1)
    //   - colloid-colloid  pairs (type_id == 2)
    //   - monomer-monomer  pairs (type_id == 0) only when mono_ev is set (excluded
    //     volume enabled); otherwise they are skipped. When enabled, WCA acts on ALL
    //     monomer pairs, including the bonded intra-dumbbell pair.
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
    double dr, dr_inv, rcut, sig, fmag, fx, fy, fz;
    std::vector<float> &rcuts = dataStruct.rcuts;
    std::vector<float> &sigmas = dataStruct.sigmas;

    // NOTE: when mm_HI==false the monomer-monomer (AA) pairs are handled by the cell
    // list (monomer_wca) and never enter vlist, so the type_id==0 branch below is inert
    // in that mode. It remains active for the mm_HI==true path, where AA pairs are still
    // enumerated into vlist and mono_ev applies their WCA here.
    for( kk = 0; kk < (Np-1); kk++ ){
        for( int pidx : vlist[kk] ){
            type_id = pair_types[pidx];

            // Skip monomer-monomer pairs unless excluded volume is enabled.
            if (type_id == 0 && !mono_ev){
                continue;
            }

            jj = id[1][pidx];

            sig    = sigmas[type_id];
            rcut   = rcuts[type_id];
            dr     = pd[3][pidx];
            dr_inv = pd[4][pidx];

            // Repulsive WCA force magnitude (0 at/beyond r_cut = 2^(1/6)*sigma).
            fmag = wca_force_mag(dr, dr_inv, sig, rcut, epsilon);
            if( fmag == 0.0 ){
                continue;
            }

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
