// Program for computing pairwise distances between particles

#include <iostream>
#include <cmath>
#include <algorithm>
#include "data.h"
#include "arrays.h"
#include "Stokes.h"

using namespace arrays;

void PolyStokes::check_dist(){
    PetscEventScope _prof(ev_check);
    // Declare variables
    int ii , jj , kk, k3, j3, type_id;
    double dx, dy, dz, dr, dr_inv, dst, dst_inv, rverl;

    std::vector<float> &rverls = dataStruct.rverls;
    for( ii = 0; ii < (pinfo.Np-1); ii++ ){ vlist[ii].clear(); }

    // Stage-2c: on >1 rank each rank computes pd/vlist ONLY for pairs whose first (row) particle
    // is a local monomer, matching the mob() column partition. mob() then reads local pd, and
    // pair_interaction() reads the local vlist (empty for non-local rows), so its monomer WCA is
    // local and its colloid contribution is a partial that RHS() reduces. Colloid-row (BB) pairs
    // are kept on every rank (none exist for a single colloid). Serial (1 rank) does all pairs.
    PetscInt m0 = 0, m1 = pinfo.Nm;
    if (mpi_size > 1) {
        PetscInt Nm = pinfo.Nm, base = Nm / mpi_size, rem = Nm % mpi_size;
        m0 = mpi_rank * base + std::min((PetscInt)mpi_rank, rem);
        m1 = m0 + base + (mpi_rank < rem ? 1 : 0);
    }

    // Iterate over the stored pairs. When mm_HI these are all pairs (AA first, then AB,
    // BB). When mm_HI is off, only AB + BB are stored (npair_stored); the monomer-monomer
    // (AA) pairs are found separately by the cell list (build_monomer_cell_list /
    // monomer_wca), so they never appear here.
    for(ii = 0; ii < pinfo.npair_stored; ii++){

        kk = id[0][ii];
        jj = id[1][ii];

        // Skip pairs whose monomer row is not local to this rank (keep colloid-row pairs).
        if (mpi_size > 1 && kk < pinfo.Nm && (kk < m0 || kk >= m1)) continue;

        k3 = consts.ndim * kk; 
        j3 = consts.ndim * jj;

        // Compute the pairwise distances
        dx = x[k3] - x[j3];
        dy = x[k3+1] - x[j3+1];
        dz = x[k3+2] - x[j3+2];
        box.minimum_image(dx, dy, dz);   // nearest image under PBC (no-op if box inactive)
        dr = sqrt( dx*dx + dy*dy + dz*dz );

        type_id = pair_types[ii];
        rverl = rverls[type_id];

        // Update the Verlet list with the pair index
        if (dr < rverl){ 
            // std::cout <<"in check_dist: " << ii << " " << kk << " " << jj << " " << dr << " " << rverl << std::endl;
            vlist[kk].push_back(ii); 
        }

        dr_inv = 1 / dr; 

        // Compute the x,y,z components of separation vectors
        pd[0][ii] = dx * dr_inv; 
        pd[1][ii] = dy * dr_inv; 
        pd[2][ii] = dz * dr_inv; 
        pd[3][ii] = dr; 
        pd[4][ii] = dr_inv;

        // UNCOMMENT below for lubrication-related calcs
        // Store the logarithms as well for lubrication calculations
        // Lubrication calculations valid for 2.1 < r < 4

        // if( dr < 4 ){

        //     dst = dr - 2; 
        //     dst_inv = 1 / dst; 

        //     pd[5][ii] = dst; 
        //     pd[6][ii] = dst_inv; 
        //     pd[7][ii] = log( dst_inv );
        // }
    }
}