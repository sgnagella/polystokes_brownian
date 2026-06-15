// Program for computing pairwise distances between particles

#include <iostream>
#include <cmath>
#include "data.h"
#include "arrays.h"
#include "Stokes.h"

using namespace arrays;

void PolyStokes::check_dist(){
    // Declare variables 
    int ii , jj , kk, k3, j3, type_id;
    int& kk_AB = id_AB[0]; 
    double dx, dy, dz, dr, dr_inv, dst, dst_inv, rverl;
    
    std::vector<float> &rverls = dataStruct.rverls;
    for( ii = 0; ii < (pinfo.Np-1); ii++ ){ vlist[ii].clear(); }

    for(ii = 0; ii < pinfo.npair; ii++){

        // Ignore dist checks if we are neglect monomer-monomer HI
        if( !mm_HI && ii < kk_AB ){
            continue;
        }

        kk = id[0][ii]; 
        jj = id[1][ii];

        k3 = consts.ndim * kk; 
        j3 = consts.ndim * jj;

        // Compute the pairwise distances
        dx = x[k3] - x[j3];
        dy = x[k3+1] - x[j3+1];
        dz = x[k3+2] - x[j3+2]; 
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