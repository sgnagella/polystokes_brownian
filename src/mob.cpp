// This routine computes the far-field contribution
// to the grand mobility matrix, which is then inverted
// Writes the per particle mobilities into Petsc Matrix objects for ZMUF, ZMUS, ZMES
// changed: zmuf, zmus, zmes

#include <iostream>
#include <cmath>
#include "arrays.h"
#include "multi_arrays.h"
#include "Stokes.h"
// #include <petscmat.h>

using namespace arrays; 

rank2_array mob_a(boost::extents[3][3]); // translation-force
rank2_array mob_b(boost::extents[3][3]); // rotation-force
rank2_array mob_bt(boost::extents[3][3]); // translation-torque ('TILDE')
rank2_array mob_c(boost::extents[3][3]); // rotation-torque
rank2_array mob_m(boost::extents[5][5]); // stress-strain
rank2_array mob_gt(boost::extents[3][5]); 
rank2_array mob_ht(boost::extents[3][5]);

rank3_array gt(boost::extents[3][3][3]); // translation-strain
rank3_array ht(boost::extents[3][3][3]); // rotation-strain
rank4_array m(boost::extents[3][3][3][3]); // stress-strain

rank2_array delta(boost::extents[3][3]);
rank3_array eps(boost::extents[3][3][3]);

void PolyStokes::mobility(
    double dr, 
    double dr_inv , 
    double dx, 
    double dy, 
    double dz, 
    bool self, 
    bool AB=false, 
    bool AA=false 
    ){
    
    // Computes the self and pair contributions to the
    // far-field mobility tensor 

    // Inputs: dr_inv - inverse distance bewteen particle centers
    // dx , dy , dz - the x , y , z distance bewteen particle centers
    // self - if true, fill self tensor; if false, fill pair.
    // AB - if true, applies size differences to translation-force HI and translation-stress HI
    // (we only consider force moment on monomers)

    // Outputs: none
    // Changes; mob_a , mob_b , mob_c , mob_bt , mob_gt , mob_ht , mob_m
    // (These are the contributions to the grand mobility tensors, either self or pair.)

    // Declare scalar mobility functions
    // x's are the parallel mobilities to the line-of-centers
    // y's are the perpendicular mobilities to the line-of-centers
    int ii, jj, kk;
    double x12a , y12a , y12b , x12c , y12c;
    double x12g , y12g , y12h , x12m , y12m, z12m;

    int& ndim = consts.ndim;
    int& const5 = consts.const5;

    // Determine the self contribution
    if( self ){

        // Independent of pairwise separation (dr_inv , dx, dy, dz)
        // Simply are the normalizations by Stokes drag
        double moba_self = 1.0; 
        double mobc_self = 3./4.;
        double mobm_self = 9./10.; 
        double mobm_self_2 = 9./5.; 

        for(ii = 0; ii < ndim; ii++){
            mob_a[ii][ii] = moba_self;
            mob_c[ii][ii] = mobc_self;
        }

        mob_m[0][4] = mobm_self; 
        mob_m[4][0] = mobm_self;

        for(ii = 0; ii < const5; ii++){
            mob_m[ii][ii] = mobm_self_2; 
        }

    }

    else{
        // Determine the pair contribution
        // Create vector to store dx dy dz 
        std::vector<double> e = {dx , dy , dz};  

        double& beta2 = pinfo.beta2;

        // Generate dyadic product of pairwise separation 
        rank2_array ee(boost::extents[ndim][ndim]); 
        initialize_rank2_array(ee);

        for(ii = 0; ii < ndim; ii++){
            ee[ii][ii] = e[ii] * e[ii];

            for(int jj = ii + 1; jj < ndim; jj++){
                ee[ii][jj] = e[ii] * e[jj]; 
                ee[jj][ii] = ee[ii][jj];
            }
        }

        // Define powers of pair separations
        double dr_inv2 = dr_inv * dr_inv; 
        double dr_inv3 = dr_inv2 * dr_inv;

        // Compute the analytical pair mobilities
        // Translation-force coupling
        if( AB ){
            x12a = dr_inv - coeffs.c1d2 * pinfo.beta2 * dr_inv3;
            y12a = coeffs.c1d2 * x12a;
        }

        else if( AA ){
            if( dr > 2.0 * pinfo.beta ){
                x12a = coeffs.c3d8 * dr_inv - 2.0 * dr_inv3;
                y12a = coeffs.c3d4 * dr_inv - coeffs.c1d2 * dr_inv3;
            }

            else{
                x12a = coeffs.c3d8 * dr - 1.0;
                y12a = 1.0 - coeffs.c9d32 * dr;
            }
        }
        else{
            x12a = coeffs.c3d2 * dr_inv - dr_inv3; 
            y12a = coeffs.c3d4 * dr_inv + coeffs.c1d2 * dr_inv3;
        }

        for(ii = 0; ii < ndim; ii++){
            // Diagonal components
            mob_a[ii][ii] = x12a * ee[ii][ii] + y12a * (1. - ee[ii][ii]); 

            // Off-diagonal components  
            for(jj = ii + 1; jj < ndim; jj++){
                mob_a[ii][jj] = x12a * ee[ii][jj] - y12a * ee[ii][jj];
                mob_a[jj][ii] = mob_a[ii][jj]; 
            }                    
        }

        if ( AA ){
            return;
        }
         
        int ll, mm;
        double dr_inv4 = dr_inv3 * dr_inv; 
        double dr_inv5 = dr_inv4 * dr_inv;

        // Rotation-force coupling
        y12b = -coeffs.c3d4 * dr_inv2;

        // Translation-stress coupling
        if( AB ){
            x12g = coeffs.c9d4 * dr_inv2 - coeffs.c9d20 * dr_inv4 * (ndim + const5 * beta2); 
            y12g = coeffs.c9d10 * dr_inv4 * (coeffs.c1d2 + coeffs.c5d6 * beta2);
        }
        else{
            x12g = coeffs.c9d4 * dr_inv2 - coeffs.c18d5 * dr_inv4;
            y12g = coeffs.c6d5 * dr_inv4; 
            
        }
        
        // Populate the per-particle mobility tensors
        // Translation-force and rotation-torque
        for(ii = 0; ii < ndim; ii++){

            // // Diagonal components
            // mob_a[ii][ii] = x12a * ee[ii][ii] + y12a * (1. - ee[ii][ii]); 

            // // Off-diagonal components
            // for(jj = ii + 1; jj < ndim; jj++ ){
            //     mob_a[ii][jj] = x12a * ee[ii][jj] - y12a * ee[ii][jj]; 
                
            //     // Mobility tensor is symmetric
            //     mob_a[jj][ii] = mob_a[ii][jj]; 

            //     // mob_c[ii][jj] = x12c * ee[ii][jj] - y12c * ee[ii][jj];
            //     // mob_c[jj][ii] = mob_c[ii][jj];
            // }

            // Translation-torque 
            for(jj = 0; jj < ndim; jj++){

                kk = ndim - ii - jj; 

                if ( kk == -1 ){ kk = 2; } 
                if ( kk ==  3 ){ kk = 0; }

                mob_b[ii][jj] = y12b * eps[ii][jj][kk] * e[kk]; 

                // Use symmetry relation of grand mobility to populate mob_bt
                mob_bt[jj][ii] = mob_b[ii][jj]; 
            }
        }

        // Populate stress-force and stress-torque mobilities
        for(kk = 0; kk < ndim; kk++){
            // std::cout << kk << std::endl;
            for(ii = 0; ii <  ndim; ii++){
                for(jj = 0; jj < ndim; jj++){

                    ll = ndim - jj - kk;
                    mm = ndim - ii - kk;

                    // Correct for "cyclic" convention of unit
                    // alternating tensor  
                    // if ( ll == -1 ){ ll = 2; }
                    // if ( ll == 3 ){ ll = 0; }
                    // if ( mm == -1 ){ mm = 2; }
                    // if ( mm == 3 ){ mm = 0; }

                    gt[kk][ii][jj] = -x12g * ( ee[ii][jj]  - coeffs.c1d3 * delta[ii][jj] )
                        * e[kk] + y12g * ( e[ii] * delta[jj][kk] + e[jj] * delta[ii][kk] - 2. * ee[ii][jj] * e[kk] );  
                }
            }
        }

        // Flatten higher order tensors onto symmetric 2nd rank tensors
        for(ii = 0; ii < ndim; ii++){
            mob_gt[ii][0] = gt[ii][0][0] - gt[ii][2][2]; 
            mob_gt[ii][1] = 2. * gt[ii][0][1]; 
            mob_gt[ii][2] = 2. * gt[ii][0][2]; 
            mob_gt[ii][3] = 2. * gt[ii][1][2];
            mob_gt[ii][4] = gt[ii][1][1] - gt[ii][2][2]; 
        }

        if( AB ){
            return;
        }

        // Rotation-torque coupling
        x12c =  coeffs.c3d4 * dr_inv3;
        y12c =  -coeffs.c3d8 * dr_inv3; 

        // Rotation-stress coupling
        y12h = -coeffs.c9d8 * dr_inv3;

        // Strain-Stress coupling
        x12m = -coeffs.c9d2 * dr_inv3 + coeffs.c54d5 * dr_inv5;
        y12m = coeffs.c9d4 * dr_inv3 - coeffs.c36d5 * dr_inv5; 
        z12m = coeffs.c9d5 * dr_inv5;

        // Colloidal contributions
        // Translation-rotation coupling
        for(ii = 0; ii < ndim; ii++){
            // Diagonal components
            mob_c[ii][ii] = x12c * ee[ii][ii] + y12c * (1. - ee[ii][ii]);

            for(jj = ii + 1; jj < ndim; jj++ ){
                mob_c[ii][jj] = x12c * ee[ii][jj] + y12c * ee[ii][jj];
                mob_c[jj][ii] = mob_c[ii][jj];
            }
        }

        // Rotation-stress coupling
        for(kk = 0; kk < ndim; kk++){
            // std::cout << kk << std::endl;
            for(ii = 0; ii <  ndim; ii++){
                for(jj = 0; jj < ndim; jj++){

                    ll = ndim - jj - kk;
                    mm = ndim - ii - kk;

                    // Correct for "cyclic" convention of unit
                    // alternating tensor  
                    if ( ll == -1 ){ ll = 2; }
                    if ( ll == 3 ){ ll = 0; }
                    if ( mm == -1 ){ mm = 2; }
                    if ( mm == 3 ){ mm = 0; }  
                    
                    ht[kk][ii][jj] = y12h * ( ee[ii][ll] * eps[jj][kk][ll] + ee[jj][mm] * eps[ii][kk][mm] );
                }
            }
        }

        // Populate stress-strain mobility
        for(ii = 0; ii < ndim; ii++){
            for(jj = 0; jj < ndim; jj++){
                for(kk = 0; kk < ndim; kk++){
                    for(ll = 0; ll < ndim; ll++){

                        m[ii][jj][kk][ll] = coeffs.c3d2 * x12m * ( ee[ii][jj] - coeffs.c1d3 * delta[ii][jj] )
                            * ( ee[kk][ll] - coeffs.c1d3 * delta[kk][ll] )
                            + coeffs.c1d2 * y12m * ( ee[ii][kk] * delta[jj][ll]
                            + ee[jj][kk] * delta[ii][ll] + ee[ii][ll] * delta[jj][kk]
                            + ee[jj][ll] * delta[ii][kk] - 4. * ee[ii][jj] * ee[kk][ll] )
                            + coeffs.c1d2 * z12m * ( delta[ii][kk] * delta[jj][ll]
                            + delta[jj][kk] * delta[ii][ll] - delta[ii][jj] * delta[kk][ll]
                            + ee[ii][jj] * delta[kk][ll] + ee[kk][ll] * delta[ii][jj]
                            - ee[ii][kk] * delta[jj][ll] - ee[jj][kk] * delta[ii][ll]
                            - ee[ii][ll] * delta[jj][kk] - ee[jj][ll] * delta[ii][kk]
                            + ee[ii][jj] * ee[kk][ll] );
                    }
                }
            }
        }

        // "Flatten" the higher order tensors onto symmetric , 2nd rank tensors w/o any loss of information
        // This segment of code converts the pair contributinos to the grand mobility tensor into a 
        // symmetric matrix. Using the following conversions of the strain rate E and the 
        // the stresslet S into the vectors EV and SV respectively. 
        // EV_1 = E_11 - E_33, EV_2 = 2 E_12 , EV_3 = 2 E_13 , EV_4 = 2 E_23 , EV_5 = E_22 - E_33
        // SV_1 = S_11 , SV_2 = S_12 = S_21 , SV_3 = S_13 = S_31, SV_4 = S_23 = S_32 , SV_5 = S_22
        
        for(ii = 0; ii < ndim; ii++){
            // mob_gt[ii][0] = gt[ii][0][0] - gt[ii][2][2]; 
            // mob_gt[ii][1] = 2. * gt[ii][0][1]; 
            // mob_gt[ii][2] = 2. * gt[ii][0][2]; 
            // mob_gt[ii][3] = 2. * gt[ii][1][2];
            // mob_gt[ii][4] = gt[ii][1][1] - gt[ii][2][2]; 

            mob_ht[ii][0] = ht[ii][0][0] - ht[ii][2][2]; 
            mob_ht[ii][1] = 2. * ht[ii][0][1]; 
            mob_ht[ii][2] = 2. * ht[ii][0][2]; 
            mob_ht[ii][3] = 2. * ht[ii][1][2];
            mob_ht[ii][4] = ht[ii][1][1] - ht[ii][2][2]; 
        }

        for(ii = 0; ii < const5; ii++){

            if( ( ii == 0 ) or ( ii == 4 ) ){
                mob_m[ii][0] = m[ mesid[0][ii] ][ mesid[0][ii] ][0][0]
                    - m[ mesid[0][ii] ][ mesid[0][ii] ][2][2]
                    - ( m[ mesid[1][ii] ][ mesid[1][ii] ][0][0] 
                    - m[ mesid[1][ii] ][ mesid[1][ii] ][2][2] );
                
                mob_m[ii][1] = 2. * ( m[ mesid[0][ii] ][ mesid[0][ii] ][0][1]  
                    - m[ mesid[1][ii] ][ mesid[1][ii] ][0][1] );
                
                mob_m[ii][2] = 2. * ( m[ mesid[0][ii] ][ mesid[0][ii] ][0][2]
                    - m[ mesid[1][ii] ][ mesid[1][ii] ][0][2] ); 

                mob_m[ii][3] = 2. * ( m[ mesid[0][ii] ][ mesid[0][ii] ][1][2] 
                    - m[ mesid[1][ii] ][ mesid[1][ii] ][1][2] ); 

                mob_m[ii][4] = m[ mesid[0][ii] ][ mesid[0][ii] ][1][1] 
                    - m[ mesid[0][ii] ][ mesid[0][ii] ][2][2]
                    - ( m[ mesid[1][ii] ][ mesid[1][ii] ][1][1] 
                    - m[ mesid[1][ii] ][ mesid[1][ii] ][2][2] ); 
            }

            else{

                mob_m[ii][0] = 2. * ( m[ mesid[0][ii] ][ mesid[1][ii] ][0][0] 
                    - m[ mesid[0][ii] ][ mesid[1][ii] ][2][2] ); 
                
                mob_m[ii][1] = 4. * ( m[ mesid[0][ii] ][ mesid[1][ii] ][0][1] ); 
                mob_m[ii][2] = 4. * ( m[ mesid[0][ii] ][ mesid[1][ii] ][0][2] ); 
                mob_m[ii][3] = 4. * ( m[ mesid[0][ii] ][ mesid[1][ii] ][1][2] );
                mob_m[ii][4] = 2. * ( m[ mesid[0][ii] ][ mesid[1][ii] ][1][1] 
                    - m[ mesid[0][ii] ][ mesid[1][ii] ][2][2] );
            }
        }


    } // end of else
    return;
}

void PolyStokes::fill_self(){
    // This routines fills the self-mobility terms of the grand mobility matrix
    // These are independent of the pair separations and can be done once at the beginning of the simulation

    // Inputs: None
    // Outputs: None
    // Changes: A (saddle point matrix)

    int ph1, ph2, ph3, ii, jj, kk, IDX1, IDX2;
    PetscInt& Nm = pinfo.Nm;
    PetscInt& Np = pinfo.Np;
    int& ndim = consts.ndim;
    PetscInt& ndimp = consts.ndimp;
    PetscInt& nc3 = consts.nc3;
    PetscInt& nm3nc6 = consts.nm3nc6;
    int& const5 = consts.const5;
    double& beta_inv = pinfo.beta_inv;

    PetscScalar VALA = 0.0;
    PetscErrorCode ierr;

    // initialize_rank2_array(mob_a);
    // initialize_rank2_array(mob_b); 
    // initialize_rank2_array(mob_c);
    // initialize_rank2_array(mob_m);
    // initialize_rank4_array(m);

    // Fill the self mobility terms
    mobility( 0., 0. , 0. , 0. , 0. , true );
    // Fill in the monomers self-contribution (F)
    for( kk = 0; kk < Nm; kk++){
        ph1 = ndimp * kk; 
        ph2 = ph1 + ndimp; 
        
        for( ii = 0; ii < ndim; ii++){
            VALA = beta_inv * (PetscScalar)mob_a[ii][ii]; 
            IDX1 = ph1 + ii; 
            ierr = MatSetValues(A, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);
        }
    }

    // Fill in the colloids self-contribution (FTS)
    for( kk = Nm; kk < Np; kk++ ){
        // std::cout << "colloidal index kk: " << kk << std::endl;
        ph1 = ndimp * kk;       // Translational modes
        ph2 = ph1 + nc3;        // Rotational modes
        ph3 = nm3nc6 + const5 * (kk-Nm);     // Stress modes

        for( ii = 0; ii < ndim; ii++){
            VALA = (PetscScalar)mob_a[ii][ii]; 
            IDX1 = ph1 + ii; 
            ierr = MatSetValues(A, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);

            VALA = (PetscScalar)mob_c[ii][ii]; 
            IDX1 = ph2 + ii; 
            ierr = MatSetValues(A, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);
        }

        for( ii = 0; ii < const5; ii++ ){
            for( jj = 0; jj < const5; jj++ ){
                VALA = (PetscScalar)mob_m[ii][jj]; 
                IDX1 = ph3 + ii; 
                IDX2 = ph3 + jj; 
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
            }
        }

    }

    return;
}

void PolyStokes::mob(){
    // This routine computes the far-field contributions to the grand mobility tensor

    // Inputs: None
    // Outputs: None
    // Changeds: A

    int idostep; 
    int ii, jj, kk, pidx;
    int ph1, ph2, ph3, ph4, ph5, ph6;
    int& Nm = pinfo.Nm;
    int& Np = pinfo.Np;
    int& ndim = consts.ndim;
    PetscInt& nc3 = consts.nc3;
    PetscInt& nm3nc6 = consts.nm3nc6;
    int& const5 = consts.const5;
    int& npair_AA = pinfo.npair_AA;
    int& npair_AB = pinfo.npair_AB;
    int& npair_BB = pinfo.npair_BB;
    PetscErrorCode ierr; 

    initialize_rank2_array(mob_a);
    initialize_rank2_array(mob_b);
    initialize_rank2_array(mob_c);
    initialize_rank2_array(mob_m);
    initialize_rank2_array(mob_gt);
    initialize_rank2_array(mob_ht);

    initialize_rank3_array(gt); 
    initialize_rank3_array(ht);
    initialize_rank4_array(m);

    initialize_rank2_array(delta);
    initialize_rank3_array(eps);

    delta[0][0] = 1.; 
    delta[1][1] = 1.;
    delta[2][2] = 1.;

    // "Clockwise" progression of indices is positive 
    eps[0][1][2] = 1.; 
    eps[1][2][0] = 1.;
    eps[2][0][1] = 1.;

    // "Anti-clockwise" progression of indices is negative
    eps[2][1][0] = -1; 
    eps[1][0][2] = -1; 
    eps[0][2][1] = -1;

    // Print values of delta and eps 
    // std::cout << "Delta: " << std::endl;
    // for(ii = 0; ii < ndim; ii++){
    //     for(jj = 0; jj < ndim; jj++){
    //         std::cout << delta[ii][jj] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Eps: " << std::endl;
    // for(ii = 0; ii < ndim; ii++){
    //     for(jj = 0; jj < ndim; jj++){
    //         for(kk = 0; kk < ndim; kk++){
    //             std::cout << eps[ii][jj][kk] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // initialize_rank2_array(mob_a);
    // Print elements of mob_a 
    // std::cout << "mob_a: " << std::endl;
    // for(ii = 0; ii < ndim; ii++){
    //     for(jj = 0; jj < ndim; jj++){
    //         std::cout << mob_a[ii][jj] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // initialize_rank2_array(mob_b); 

    // Print elements of mob_a 
    // std::cout << "mob_b: " << std::endl;
    // for(ii = 0; ii < ndim; ii++){
    //     for(jj = 0; jj < ndim; jj++){
    //         std::cout << mob_b[ii][jj] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // initialize_rank2_array(mob_c); 
    // Print elements of mob_a 
    // std::cout << "mob_c: " << std::endl;
    // for(ii = 0; ii < ndim; ii++){
    //     for(jj = 0; jj < ndim; jj++){
    //         std::cout << mob_c[ii][jj] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // initialize_rank2_array(mob_m); 
    // Print elements of mob_m 
    // for(ii = 0; ii < const5; ii++){
    //     for(jj = 0; jj < const5; jj++){
    //         std::cout << mob_m[ii][jj] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // initialize_rank2_array(mob_gt);
    // initialize_rank2_array(mob_ht);

    // initialize_rank3_array(gt); 
    // initialize_rank3_array(ht);
    // initialize_rank4_array(m);

    // std::cout << "Creating matrix..." << std::endl;

    // Fill the self mobility terms 
    // Note that in this formulation, matrix is filled by particle index
    // So the corresponding vector of FTE is F1, T1, E1 , F2 , T2 , E2 , etc.
    // mobility( 0. , 0. , 0. , 0. , true );

    // std::cout << "Computing the far-field contributions to the grand mobility tensor" << std::endl;

    PetscScalar VALA = 0.0; 
    PetscInt IDX1, IDX2;

    // // Fill in the monomers self-contribution (F)
    // for( kk = 0; kk < Nm; kk++){
    //     ph1 = ndimp * kk; 
    //     ph2 = ph1 + ndimp; 
        
    //     for( ii = 0; ii < ndim; ii++){
    //         VALA = beta_inv * (PetscScalar)mob_a[ii][ii]; 
    //         IDX1 = ph1 + ii; 
    //         ierr = MatSetValues(A, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);
    //     }
    // }

    // // Fill in the colloids self-contribution (FTS)
    // for( kk = Nm; kk < Np; kk++ ){
    //     // std::cout << "colloidal index kk: " << kk << std::endl;
    //     ph1 = ndimp * kk;       // Translational modes
    //     ph2 = ph1 + nc3;        // Rotational modes
    //     ph3 = nm3nc6 + const5 * (kk-Nm);     // Stress modes

    //     for( ii = 0; ii < ndim; ii++){
    //         VALA = (PetscScalar)mob_a[ii][ii]; 
    //         IDX1 = ph1 + ii; 
    //         ierr = MatSetValues(A, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);

    //         VALA = (PetscScalar)mob_c[ii][ii]; 
    //         IDX1 = ph2 + ii; 
    //         ierr = MatSetValues(A, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);
    //     }

    //     for( ii = 0; ii < const5; ii++ ){
    //         for( jj = 0; jj < const5; jj++ ){
    //             VALA = (PetscScalar)mob_m[ii][jj]; 
    //             IDX1 = ph3 + ii; 
    //             IDX2 = ph3 + jj; 
    //             ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
    //         }
    //     }

    // }


    // std::cout << "last index IDX1 " << IDX1 << std::endl;
    // for( kk = 0; kk < Np ; kk++){

    //     ph1 = ndim * kk; // Translational modes
    //     ph2 = ph1 + n3;  // Rotational modes
    //     ph3 = 5 * kk;    // Stress modes

    //     for( ii = 0; ii < ndim; ii++ ){
    //         VALA = (PetscScalar)mob_a[ii][ii];
    //         IDX1 = ph1 + ii;
    //         MatSetValues(M, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES);
    //         // assign_per_particle_to_matrix(&IDX1, &IDX1, &VALA, ph1 + ii, ph1 + ii, mob_a, ZMUF);
 
    //         VALA = (PetscScalar)mob_c[ii][ii];
    //         IDX1 = ph2 + ii;
    //         MatSetValues(M, 1, &IDX1, 1, &IDX1, &VALA, INSERT_VALUES);
    //         // assign_per_particle_to_matrix(&IDX1, &IDX1, &VALA, ph2 + ii, ph2 + ii, mob_c, ZMUF);
    //     }

    //     for( ii = 0; ii < const5 ; ii++ ){
            
    //         for( jj = 0; jj < const5; jj++ ){
                
    //             VALA = (PetscScalar)mob_m[ii][jj];
    //             IDX1 = ph3 + ii + n6p;
    //             IDX2 = ph3 + jj + n6p;
    //             MatSetValues(M, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES);
    //             // assign_per_particle_to_matrix(&IDX1, &IDX1, &VALA, ph3 + ii, ph3 + jj, mob_m, ZMES);
    //         }
    //     }
    // }

    // Fill in pair mobility terms
    // Monomer-monomer interactions
    if(mm_HI){
        // std::cout<< "Applying monomer-monomer HI" << std::endl;
        for( kk = 0; kk < npair_AA; kk++){

            // If only considering inter-chain monomer HI
            // Check the pair belongs to different chains
            if(chain_HI && id_AA[kk][1]){
                continue;
            }

            pidx = id_AA[kk][0];
            ph1 = id[0][pidx];
            ph2 = id[1][pidx];
            
            // If only considering chain-chain HI
            // Pair mobility interaction is 0 for monomers of the same chain
            // if(chain_HI && (chain_ids[ph1] == chain_ids[ph2])){
            //     continue;
            // }

            mobility( pd[3][pidx], pd[4][pidx] , pd[0][pidx] , pd[1][pidx] , pd[2][pidx] , false, false, true); 
            
            // TODO: account for only chain-chain HI 
            // along the lines of 
            // chain_id_1 = cid[id[0][pidx]]
            // chain_id_2 = cid[id[0][pidx]]
            // cid is size(N_mono_total) array
            // each monomer is assigned to a chain
            // if chain_id_1 == chain_id_2 -> continue

            // ph1 = ndim * id[0][pidx]; 
            // ph2 = ndim * id[1][pidx];

            ph1 *= ndim;
            ph2 *= ndim;
    
            for( ii = 0; ii < ndim ; ii++ ){
    
                for( jj = 0; jj < ndim; jj++ ){ 
    
                    IDX1 = ph1 + ii;
                    IDX2 = ph2 + jj;
                    VALA = (PetscScalar)mob_a[ii][jj];
                    // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_a VALA " << VALA << std::endl;
                    ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
    
                    IDX1 = ph2 + ii; 
                    IDX2 = ph1 + jj;
                    VALA = (PetscScalar)mob_a[jj][ii];
                    // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_a VALA " << VALA << std::endl;
                    ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                }
            }
        }
    }
    
    // Monomer-colloid interactions
    for( kk = 0; kk < npair_AB ; kk++){
        pidx = id_AB[kk];
        // Compute the mobilities for each particle pair
        mobility( pd[3][pidx], pd[4][pidx] , pd[0][pidx] , pd[1][pidx] , pd[2][pidx] , false, true, false); 
        
        ph1 = id[0][pidx]; // monomer index
        ph2 = id[1][pidx]; // colloid index

        // std::cout << "in AB interactions ph2 " << ph2 << std::endl;

        // ph5 = nm3nc6 + const5 * ( ph1 - Nm ); 
        ph6 = const5 * ( ph2 - Nm ) + nm3nc6;

        // std::cout << "in AB interactions ph6 " << ph6 << std::endl;

        ph1 = ndim * ph1; 
        ph2 = ndim * ph2;

        // ph1 = 6 * ( ph1 );
        // ph2 = 6 * ( ph2 );

        ph4 = ph2 + nc3; 

        // std::cout << "IN AB_MC ph4 " << ph4 << std::endl;

        for( ii = 0; ii < ndim ; ii++ ){

            for( jj = 0; jj < ndim; jj++ ){ 

                IDX1 = ph1 + ii;
                IDX2 = ph2 + jj;
                VALA = (PetscScalar)mob_a[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_a VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                
                IDX1 = ph4 + ii;
                IDX2 = ph1 + jj;
                VALA = (PetscScalar)mob_b[ii][jj]; 
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_b VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph1 + ii; 
                IDX2 = ph4 + jj;
                // std::cout<< "in AB_MC IDX2 " << IDX2 << std::endl;
                VALA = (PetscScalar)mob_bt[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_bt VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                // Symmetric contributions
                // Only have force-velocity coupling
                // Monomers have no angular velocities or torques
                IDX1 = ph2 + ii;
                IDX2 = ph1 + jj;
                VALA = (PetscScalar)mob_a[jj][ii];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_a VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
            }

            // velocity-stress couplings
            for( jj = 0; jj < const5; jj++ ){
                IDX1 = ph1 + ii;
                IDX2 = ph6 + jj;
                VALA = (PetscScalar)mob_gt[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_gt VALA " << VALA << std::endl;
                // std::cout << "in US_MC IDX2 " << IDX2 << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                ierr = MatSetValues(A, 1, &IDX2, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);
                // No stresses on the monomers, so no need to populate the symmetric terms
            }
        }
    }

    for( kk = 0; kk < npair_BB; kk++){
        pidx = id_BB[kk];
        // Compute the mobilities for each particle pair
        mobility( pd[3][pidx], pd[4][pidx] , pd[0][pidx] , pd[1][pidx] , pd[2][pidx] , false, false);
        ph1 = id[0][pidx]; // colloid index
        ph2 = id[1][pidx]; // colloid index

        ph5 = nm3nc6 + const5 * ( ph1 - Nm ); 
        ph6 = nm3nc6 + const5 * ( ph2 - Nm );

        ph1 = ndim * ph1; 
        ph2 = ndim * ph2;

        ph3 = ph1 + nc3; 
        ph4 = ph2 + nc3; 

        for( ii = 0; ii < ndim; ii++){
            for( jj = 0; jj < ndim; jj++){
                
                IDX1 = ph1 + ii;
                IDX2 = ph2 + jj;
                VALA = (PetscScalar)mob_a[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_a VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph3 + ii;
                VALA = (PetscScalar)mob_b[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_b VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph2 + ii;
                IDX2 = ph3 + jj; 
                VALA = (PetscScalar)mob_bt[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_bt VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph3 + ii;
                IDX2 = ph4 + jj;
                VALA = (PetscScalar)mob_c[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_c VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                // Symmetric contributions
                IDX1 = ph2 + ii;
                IDX2 = ph1 + jj;
                VALA = (PetscScalar)mob_a[jj][ii];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_a VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph4 + ii;
                VALA = -(PetscScalar)mob_b[jj][ii];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_b VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph1 + ii;
                IDX2 = ph4 + jj;
                VALA = -(PetscScalar)mob_bt[jj][ii];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_bt VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph4 + ii;
                IDX2 = ph3 + jj; 
                VALA = (PetscScalar)mob_c[jj][ii];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_c VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
            }

            for( jj = 0; jj < const5; jj++){
                
                IDX1 = ph1 + ii;
                IDX2 = ph6 + jj;
                VALA = (PetscScalar)mob_gt[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_gt VALA " << VALA << std::endl;
                // 1-2 interaction
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                ierr = MatSetValues(A, 1, &IDX2, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);
                
                IDX1 = ph2 + ii;
                IDX2 = ph5 + jj;
                VALA *= -1.0;
                // 2-1 interaction
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_gt VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                ierr = MatSetValues(A, 1, &IDX2, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph3 + ii;
                IDX2 = ph6 + jj;
                VALA = (PetscScalar)mob_ht[ii][jj];
                // 1-2 interaction
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_ht VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                ierr = MatSetValues(A, 1, &IDX2, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);

                IDX1 = ph4 + ii;
                IDX2 = ph5 + jj;
                // 2-1 interaction
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_ht VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                ierr = MatSetValues(A, 1, &IDX2, 1, &IDX1, &VALA, INSERT_VALUES); CHKERRV(ierr);

            }
        }

        for( ii = 0; ii < const5; ii++){
            for( jj = 0; jj < const5; jj++){
                IDX1 = ph5 + ii;
                IDX2 = ph6 + jj;
                VALA = (PetscScalar)mob_m[ii][jj];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_m VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
                
                IDX1 = ph6 + ii;
                IDX2 = ph5 + jj;
                VALA = (PetscScalar)mob_m[jj][ii];
                // std::cout << "IDX1 " << IDX1 << " IDX2 " << IDX2 << " mob_m T VALA " << VALA << std::endl;
                ierr = MatSetValues(A, 1, &IDX1, 1, &IDX2, &VALA, INSERT_VALUES); CHKERRV(ierr);
            }
        }
    }
    // std::cout << "Last index IDX2 " << IDX2 << std::endl;
    // // Check for symmetry
    // PetscBool isSymmetric;
    // PetscReal tol = 1e-10; // Tolerance for comparison

    // Assemble the saddle point matrix matrix
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);

    // View the matrix
    // MatView(M, PETSC_VIEWER_STDOUT_SELF);

    // Check for symmetry
    // ierr = MatIsSymmetric(A, tol, &isSymmetric); CHKERRV(ierr);

    // // Print result
    // if (isSymmetric == PETSC_TRUE) {
    //     PetscPrintf(PETSC_COMM_WORLD, "The matrix A is symmetric.\n");
    // } else {
    //     PetscPrintf(PETSC_COMM_WORLD, "The matrix A is not symmetric.\n");
    // }

    // Save matrix to file

    // if( timeinfo.t == timeinfo.dt ){
    //     PetscViewer viewer;
    //     PetscViewerASCIIOpen(PETSC_COMM_WORLD, "saddle.csv", &viewer);
    //     PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE); // Forces dense output
    //     MatView(A, viewer);
    //     PetscViewerPopFormat(viewer);
    //     PetscViewerDestroy(&viewer);
    // }

    return;
}