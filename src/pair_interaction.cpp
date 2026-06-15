#include "arrays.h"
#include "data.h"
#include "params.h"
#include "pair_interaction.h"
#include <iostream>
#include <cmath>

using namespace arrays;

void pair_interaction(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, Data& dataStruct){
    // Computes pair interaction using Verlet list 

    int ii, jj, kk, ll, mm, jj3, kk3, k3, l3, type_id;
    PetscInt& ndimp = consts.ndimp;
    PetscInt& Np = pinfo.Np;
    PetscScalar r[ndimp+1];
    double& tau = pinfo.tau;
    double& c50d49p49 = consts.c50d49p49;
    double& c49d50 = consts.c49d50;
    double dr, force, fx, fy, fz, ext, ext2, rcut, dr_inv;
    double sig, sig2, sig4, sig6, sig12, sig49, sig50;
    std::vector<float> &rcuts = dataStruct.rcuts;
    std::vector<float> &sigmas = dataStruct.sigmas;

    for( kk = 0; kk < (Np-1); kk++ ){
        for( int pidx : vlist[kk] ){
            type_id = pair_types[pidx]; 

            if (type_id == 0){
                continue;
            }
            
            jj = id[1][pidx];

            
            sig = sigmas[type_id];
            rcut = rcuts[type_id];
            dr = pd[3][pidx];
            dr_inv = pd[4][pidx];

            // unit vectors 
            fx = pd[0][pidx];
            fy = pd[1][pidx];
            fz = pd[2][pidx];

            kk3 = ndimp * kk; 
            jj3 = ndimp * jj;

            // sig50 = std::pow( sig * dr_inv , 12);
            // sig2 = sig * sig * dr_inv * dr_inv;
            // sig4 = sig2 * sig2;
            // sig6 = sig4 * sig2;
            // sig12 = sig6 * sig6;
            // force = 24. * dr_inv * (2. * sig12 - sig6);
            
            // fx *= force;
            // fy *= force;
            // fz *= force; 
        
            if ( type_id == 1 ){

                if(dr < rcut){
                    // std::cout << "force: " << force << std::endl;
                    // std::cout << "dr: " << dr << std::endl;
                    // std::cout << "sig: " << sig << std::endl;
                    // std::cout << "radii[kk]: " << radii[kk] << std::endl;
                    // std::cout << "radii[jj]: " << radii[jj] << std::endl;
                    // std::cout << "type id: " << type_id << std::endl;

                    // force = dr - (radii[kk] + radii[jj] + 0.001);
                    // force = std::exp( -tau * force );
                    // force = force/(1-force);
                    // fx *= force;
                    // fy *= force;
                    // fz *= force;
                    // fext[kk3] -= fx;
                    // fext[kk3+1] -= fy;
                    // fext[kk3+2] -= fz;

                    // Melrose-Heyes calc
                    // force = sig - dr;

                    // tau potential
                    // force = dr - (sig);
                    // force = std::exp( -tau * force );
                    // force = force/(1-force);
                    // fx *= force;
                    // fy *= force;
                    // fz *= force;
                    // fext[kk3] -= fx;
                    // fext[kk3+1] -= fy;
                    // fext[kk3+2] -= fz;

                    // 50-49 LJ
                    // sig50 = std::pow( sig * dr_inv , 50);
                    // sig49 = std::pow( sig * dr_inv , 49);
                    // force = c50d49p49 * dr_inv * (-sig50 + c49d50 * sig49);

                    // 12-6 LJ (WCA)
                    // sig50 = std::pow( sig * dr_inv , 12);
                    // force = 48.0 * dr_inv * sig50

                    // fext[kk3] -= fx;
                    // fext[kk3+1] -= fy;
                    // fext[kk3+2] -= fz;


                    // UNCOMMENT below for original Melrose-Heyes
                    // We take the overlap corrections to be small 
                    // that they unaffect mobility calcs ->
                    // they are accounted in force calcs later

                    if( dr < sig ){
                        // std::cout << "dr: " << dr << std::endl;
                        // std::cout << "sig: " << sig << std::endl;
                        // std::cout << "radii[kk]: " << radii[kk] << std::endl;
                        // std::cout << "radii[jj]: " << radii[jj] << std::endl;
                        // std::cout << "type id: " << type_id << std::endl;
                        force = sig - dr;
                        fx *= force;
                        fy *= force;
                        fz *= force;
                        x[kk3] += fx;
                        x[kk3+1] += fy;
                        x[kk3+2] += fz;
                    }
                    
                    
                    // UNCOMMENT below for force version
                    // Implement through extensional force
                    // Loop through the bond ids and compute 
                    // the additional force on the monomers
                    // approximate force is given by current extension * (rcut-dr) 
                    // for( ii = 0; ii < nbonds; ii++){
                    //     if( arrays::bond_list[kk][ii] == 1 ){
                    //         // Compute the force
                    //         ll = bond_ids[1][ii];
                    //         k3 = ndim * kk; 
                    //         l3 = ndim * ll;
                            
                    //         // current extension
                    //         r[0] = x[k3] - x[l3];
                    //         r[1] = x[k3 + 1] - x[l3 + 1];
                    //         r[2] = x[k3 + 2] - x[l3 + 2];
                    //         r[3] = sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] );
                            
                    //         ext = -kbond * (1 - (r0/r[3]));         // harmonic extension
                    //         ext *= 1/(1-(r[3]/Lmax)*(r[3]/Lmax));    // fene correction
                            
                    //         fx *= ext;
                    //         fy *= ext;
                    //         fz *= ext;

                    //         fext[kk3] -= fx;
                    //         fext[kk3+1] -= fy;
                    //         fext[kk3+2] -= fz;
                    //     }
                    // }
                }
            }

            else{
                
                if(dr < rcut){
                    // std::cout << "force: " << force << std::endl;
                    // std::cout << "dr: " << dr << std::endl;
                    // std::cout << "sig: " << sig << std::endl;
                    // std::cout << "radii[kk]: " << radii[kk] << std::endl;
                    // std::cout << "radii[jj]: " << radii[jj] << std::endl;
                    // std::cout << "type id: " << type_id << std::endl;   

                    // force = (dr - sig);
                    // force = std::exp( -tau * force );
                    // force = force/(1-force);

                    // sig50 = std::pow( sig * dr_inv , 50);
                    // sig49 = std::pow( sig * dr_inv , 49);
                    // force = c50d49p49 * dr_inv * (-sig50 + c49d50 * sig49);

                    // sig50 = std::pow( sig * dr_inv , 12);
                    // force = 48.0 * dr_inv * sig50

                    // if( force > 1e2 ){
                    //     std::cout << "force: " << force << std::endl;
                    //     std::cout << "dr: " << dr << std::endl;
                    //     std::cout << "radii[kk]: " << radii[kk] << std::endl;
                    //     std::cout << "radii[jj]: " << radii[jj] << std::endl;
                    // }

                    // fx *= force;
                    // fy *= force;
                    // fz *= force;
                    
                    // x[kk3] += fx;
                    // x[kk3+1] += fy;
                    // x[kk3+2] += fz;
                    // x[jj3] -= fx;
                    // x[jj3+1] -= fy;
                    // x[jj3+2] -= fz;

                    // fext[kk3] -= fx;
                    // fext[kk3+1] -= fy;
                    // fext[kk3+2] -= fz;
                    // fext[jj3] += fx;
                    // fext[jj3+1] += fy;
                    // fext[jj3+2] += fz;

                    if( dr < sig ){
                        // std::cout << "dr: " << dr << std::endl;
                        // std::cout << "sig: " << sig << std::endl;
                        // std::cout << "radii[kk]: " << radii[kk] << std::endl;
                        // std::cout << "radii[jj]: " << radii[jj] << std::endl;
                        // std::cout << "type id: " << type_id << std::endl;

                        // melrose heyes
                        force = sig - dr;
                        fx *= force;
                        fy *= force;
                        fz *= force;
                        x[kk3] += fx;
                        x[kk3+1] += fy;
                        x[kk3+2] += fz;
                        x[jj3] -= fx;
                        x[jj3+1] -= fy;
                        x[jj3+2] -= fz;

                    }

                }
                

            }
        }
    }
}