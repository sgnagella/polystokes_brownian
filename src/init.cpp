// This program controls the flow of the program initialization. 
// Inputs: none
// Output: none
// Changes: none

// #include "init.h"
#include "data.h"
#include "multi_arrays.h"
#include "arrays.h"
// #include "lubdat.h"
// #include "saddle.h"
// #include "globals.h"
// #include "config.h"

#include <iostream>
#include <fstream>
#include <cmath>

#include "Stokes.h"
// #include <petscmat.h>

using namespace std;
using namespace arrays;


void PolyStokes::init_solver(){
    // Get processor rank and size
    PetscErrorCode ierr;

    // PetscMPIInt rank, size;
    // MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    // MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Message from rank %d of %d\n", rank, size);
    // PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    // Initialize the solver
    std::cout << "..." << std::endl;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRV(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRV(ierr);
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRV(ierr);
    ierr = KSPGetPC(ksp, &pc); CHKERRV(ierr);

    // uncomment to use custom preconditioner
    std::cout << "Preconditioner set" << std::endl;
    ierr = PCSetType(pc, PCFIELDSPLIT); CHKERRV(ierr);

    if(mm_HI){
        // uncomment below to use explicit shcur field splitting 
        IS is_F, is_V; 

        PetscInt idxf[consts.nm3nc11];
        PetscInt idxvel[consts.nm3nc6];

        std::iota(idxf, idxf + consts.nm3nc11, 0);
        std::iota(idxvel, idxvel + consts.nm3nc6, consts.nm3nc11);
        ierr = ISCreateGeneral(PETSC_COMM_WORLD, consts.nm3nc11, idxf, PETSC_COPY_VALUES, &is_F); CHKERRV(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_WORLD, consts.nm3nc6, idxvel, PETSC_COPY_VALUES, &is_V); CHKERRV(ierr);

        PCFieldSplitSetIS(pc, "F", is_F);   
        PCFieldSplitSetIS(pc, "V", is_V);    

        // ierr = PCFieldSplitSetType(pc, PC_COMPOSITE_GKB); CHKERRV(ierr);
        PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);  // or PC_COMPOSITE_MULTIPLICATIVE
        PCFieldSplitSchurPrecondition(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, NULL);  // Or PC_FIELDSPLIT_SCHUR_PRE_A11

        // Clean up index sets
        ierr = ISDestroy(&is_F); CHKERRV(ierr);
        ierr = ISDestroy(&is_V); CHKERRV(ierr);

    }
    else{
        // Detect saddle point structure
        ierr = PCFieldSplitSetDetectSaddlePoint(pc, PETSC_TRUE); CHKERRV(ierr);
    }

    std::cout << "Set operators" << std::endl;
    ierr = KSPSetFromOptions(ksp); CHKERRV(ierr);
    KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);

    return;
}

void alok_arrays(ParticleInfo& pinfo, Consts& consts){
    // Allocate memory for array operations

    // TODO: Change x and x_0 to be size n6 and correct the indexing elsewhere
    std::cout << "Allocating memory for arrays..." << std::endl;
    initialize_radii( pinfo.Np );
    initialize_x0( consts.nm3nc3 ); 
    initialize_up( consts.nm3nc6 );
    initialize_fext( consts.nm3nc6 );
    // initialize_q( consts.nc4 );
    
    std::cout << "Bookeeping arrays..." << std::endl;
    initialize_pd( consts.pd_rows, pinfo.npair); 
    initialize_id( consts.id_rows, pinfo.npair);
    initialize_pair_types( pinfo.npair );
    initialize_vlist(pinfo.Np-1);
    initialize_bond_list(pinfo.Np, pinfo.nbonds);
    initialize_chainid(pinfo.Nm);
    initialize_id_AA( pinfo.npair_AA , consts.id_rows); 
    initialize_id_AB( pinfo.npair_AB ); 
    initialize_id_BB( pinfo.npair_BB );
    initialize_bondid( consts.id_rows, pinfo.nbonds); 
    initialize_mesid( consts.id_rows, consts.const5);

    std::cout << "PETSC arrays ..." << std::endl;
    initialize_B(consts.nm3nc11, consts.nm3nc6);
    std::cout << "A" << std::endl;
    initialize_A(consts.nm6nc17, consts.nm3nc11, consts.nm3nc6);
    std::cout << "rhs" << std::endl;
    initialize_rhs(consts.nm6nc17);
    std::cout << "X" << std::endl;
    initialize_X(consts.nm6nc17);

    return;
}

void set_vars(ParticleInfo& pinfo, Data& dataStruct, Consts& consts){
    // Define other useful variables for the calculations
    int ii, jj, kk, kk_AA, kk_AB, pidx; 

    int& Nm = pinfo.Nm;
    int& Np = pinfo.Np;
    int& Nc = pinfo.Nc;
    int& Npoly = pinfo.Npoly;
    int& npair = pinfo.npair;
    int& npair_AA = pinfo.npair_AA;
    int& npair_AB = pinfo.npair_AB;
    int& npair_BB = pinfo.npair_BB;
    int& nbonds_per_poly = pinfo.nbonds_per_poly;
    int& Nmono_per_chain = pinfo.Nmono_per_chain;
    int& nbonds = pinfo.nbonds;
    int& ndim = consts.ndim;

    // Set up particle pairs by populating array "id"
    // Read the particle pairs going down a column
    // Define particle pairs and record type index of interaction
    kk = 0;
    std::cout << "Setting up chain ids" << std::endl;
    for( ii = 0; ii < Npoly; ii++){
        for( jj = 0; jj < Nmono_per_chain; jj++){
            kk = ii*Nmono_per_chain + jj; 
            chain_ids[kk] = ii; 
        }
    }

    // Print the chain ids
    for( ii = 0; ii < Nm; ii++){
        std::cout << "chain_ids[" << ii << "]= " << chain_ids[ii] << std::endl;
    }

    if( (kk+1) != Nm ){
        cout << "Number of chain ids does not match number of monomers " << std::endl;
    }

    std::cout << "Setting up particle pairs..." << std::endl;
    kk = 0; 
    // 2nd index on id_AA records whether the pair belongs to the same chain
    for(ii = 0; ii < Nm; ii++){
        for(jj = ii+1; jj < Nm; jj++){
            id[0][kk] = ii; 
            id[1][kk] = jj; 
            id_AA[kk][0] = kk; 

            if(chain_ids[ii] == chain_ids[jj]){
                id_AA[kk][1] = 1;
            }

            kk += 1;
        }
    }

    kk_AA = kk; 
    for(ii = 0; ii < Nm; ii++){
        for(jj = Nm; jj < Np; jj++){
            id[0][kk] = ii; 
            id[1][kk] = jj; 
            pair_types[kk] = 1;
            id_AB[kk-kk_AA] = kk;
            kk +=1;
        }
    }

    kk_AB = kk; 
    for(ii = Nm; ii < Np; ii++){
        for(jj = ii+1; jj < Np; jj++){
            id[0][kk] = ii;
            id[1][kk] = jj;
            pair_types[kk] = 2;
            id_BB[kk-kk_AB] = kk;
            kk += 1;
        }
    }


    if( kk != npair){
        cout << "Number of pairs populated not equal to expected value" << endl;
    }

    // Set bond ids
    std::cout << "Setting up bond ids..." << std::endl;
    kk = 0; 
    for( ii = 0; ii < Npoly; ii++ ){
        for( jj = 0; jj < nbonds_per_poly; jj++ ){
            pidx = ii * Nmono_per_chain + jj; 
            bond_ids[0][kk] = pidx; 
            bond_ids[1][kk] = pidx + 1;
            kk += 1; 
        }
    }

    if( kk != nbonds){
        cout << "Number of bonds populated not equal to expected value" << endl;
    }


    std::cout << "Setting up bond list..." << std::endl;
    // Write the bond list to go from monomer to bond ids 
    for( kk = 0; kk < nbonds; kk++ ){
        ii = bond_ids[0][kk];
        bond_list[ii][kk] = 1;
    }

    for( ii = 0; ii < Nm; ii++ ){ radii[ii] = 0.1; } // TODO: use the global parameter 'beta' for now hard-coded
    for( ii = 0; ii < Nc; ii++ ){ radii[Nm + ii] = 1.0; }

    // Set up neighborlist data
    // dataStruct = new Data;
    // dataStruct.box = {42, 42, 5};
    double prefactor = std::pow(2, 1.0/6.0);
    // double prefactor = 50./49.;
    dataStruct.rverls = {0.05 + 2*pinfo.beta, 1.5 + pinfo.beta, 2.5};       // Verlet search cutoffs for AA, AB, BB interactions
    dataStruct.sigmas = {2.*pinfo.beta, 1. + pinfo.beta, 2.0}; 
    dataStruct.rcuts = {
        prefactor*dataStruct.sigmas[0], 
        prefactor*dataStruct.sigmas[1], 
        prefactor*dataStruct.sigmas[2]
    };   // Interaction cutoff distances
    
    // dataStruct -> nlist.resize(Np, 0);

    // Populate mesid 
    mesid[0][0] = 0; 
    mesid[1][0] = 2; 

    mesid[0][1] = 0; 
    mesid[1][1] = 1; 

    mesid[0][2] = 0; 
    mesid[1][2] = 2; 

    mesid[0][3] = 1; 
    mesid[1][3] = 2;

    mesid[0][4] = 1; 
    mesid[1][4] = 2;


    return; 
}



void PolyStokes::init(){
    std::cout << "Initializing the program..." << std::endl;
    PetscInitialize(NULL, NULL, NULL, NULL);
    PetscPushErrorHandler(PetscAbortErrorHandler, NULL);

    KSP ksp;
    PC pc;

    std::cout << "Allocating memory..." << std::endl;
    alok_arrays(pinfo, consts);

    std::cout << "Setting vars..." << std::endl;
    set_vars(pinfo, dataStruct, consts);

    std::cout << "Initializing the solver..." << std::endl;
    init_solver();
    
}

