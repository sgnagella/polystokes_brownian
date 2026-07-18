#include "arrays.h"
#include "data.h"
#include "params.h"
#include "pair_interaction.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

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

void pair_interaction_hs(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev, double dt){
    // Drop-in alternative to pair_interaction() using the asymmetric-harmonic hard-sphere
    // repulsion (see hs_force_mag in the header): a linear F = k*(sig_eff - r) that acts only
    // on overlapping pairs, with k = 1/dt so the correction is O(overlap) per step regardless
    // of dt. Unlike WCA (too soft: residual overlaps make the mobility non-positive-definite)
    // and the high-exponent LJ (too stiff: force diverges and overshoots), this pushes
    // overlapping pairs apart by ~the overlap each step without blowing up. Pair enumeration,
    // the mono_ev gating, and the fext sign convention match pair_interaction().

    int kk, jj, kk3, jj3, type_id;
    PetscInt& ndimp = consts.ndimp;
    PetscInt& Np = pinfo.Np;
    double dr, sig, fmag, fx, fy, fz;
    std::vector<float> &sigmas = dataStruct.sigmas;

    // Spring constant k = kpref/dt. The nominal "remove the overlap in one step" value is
    // kpref=1, but explicit integration of F=k*delta is stable only while k*mu_eff*dt < 2, and
    // the colloid-monomer relative mobility mu_eff is O(few) (not 1), so kpref=1 sits near the
    // threshold and DIVERGES in dense/soft configs (swept: kpref=1 blew up to O(1e12) where
    // kpref=0.3 kept every eigenvalue |lambda|<0.4). Scaling k UP (e.g. 1/(dt*beta), kpref~100)
    // overshoots and is far worse -- the mobile monomer needs a SMALLER spring, not a larger
    // one. kpref=0.3 was robust (stable in every config tested, best in the hard ones, only
    // marginally deeper than kpref=1 in mild ones), so it is the default.
    // Two env knobs (read once) allow re-tuning without recompiling: HS_KPREF is the multiplier;
    // HS_KMODE=1 switches to the 1/beta scaling (k = kpref/(dt*beta)) for experimentation.
    static const double kpref = [](){ const char* s = getenv("HS_KPREF"); return s ? atof(s) : 0.3; }();
    static const int    kmode = [](){ const char* s = getenv("HS_KMODE"); return s ? atoi(s) : 0;   }();
    const double k = (kmode == 1) ? kpref / (dt * pinfo.beta) : kpref / dt;

    for( kk = 0; kk < (Np-1); kk++ ){
        for( int pidx : vlist[kk] ){
            type_id = pair_types[pidx];

            // Skip monomer-monomer pairs unless excluded volume is enabled.
            if (type_id == 0 && !mono_ev){
                continue;
            }

            jj = id[1][pidx];

            sig = sigmas[type_id];
            dr  = pd[3][pidx];

            // Repulsive harmonic force magnitude (0 unless the pair overlaps, r < sig_eff).
            fmag = hs_force_mag(dr, sig, k);
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

            // fext stores the negative of the physical force (cf. pair_interaction).
            fext[kk3]   -= fx;
            fext[kk3+1] -= fy;
            fext[kk3+2] -= fz;
            fext[jj3]   += fx;
            fext[jj3+1] += fy;
            fext[jj3+2] += fz;
        }
    }
}

void pair_interaction_highexp(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct, bool mono_ev){
    // Drop-in alternative to pair_interaction() that replaces the WCA excluded-volume force
    // with a purely-repulsive high-exponent (n=48) Lennard-Jones force (see highexp_force_mag
    // in the header). WCA is too soft here: residual colloid-monomer overlaps leave the
    // mobility operator non-positive-definite (O(0.1-1) negative eigenvalues under the sqrt).
    // The much steeper high-exponent core suppresses those overlaps. Pair enumeration, the
    // mono_ev gating, and the fext sign convention are identical to pair_interaction(); only
    // the force law and its (self-contained) cutoff differ.

    int kk, jj, kk3, jj3, type_id;
    PetscInt& ndimp = consts.ndimp;
    PetscInt& Np = pinfo.Np;
    double& epsilon = pinfo.epsilon;
    double dr, dr_inv, sig, fmag, fx, fy, fz;
    std::vector<float> &sigmas = dataStruct.sigmas;

    for( kk = 0; kk < (Np-1); kk++ ){
        for( int pidx : vlist[kk] ){
            type_id = pair_types[pidx];

            // Skip monomer-monomer pairs unless excluded volume is enabled.
            if (type_id == 0 && !mono_ev){
                continue;
            }

            jj = id[1][pidx];

            sig    = sigmas[type_id];
            dr     = pd[3][pidx];
            dr_inv = pd[4][pidx];

            // Repulsive high-exponent force magnitude (0 at/beyond r = sig_eff = 1.1*sigma).
            fmag = highexp_force_mag(dr, dr_inv, sig, epsilon);
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

            // fext stores the negative of the physical force (cf. pair_interaction).
            fext[kk3]   -= fx;
            fext[kk3+1] -= fy;
            fext[kk3+2] -= fz;
            fext[jj3]   += fx;
            fext[jj3+1] += fy;
            fext[jj3+2] += fz;
        }
    }
}
