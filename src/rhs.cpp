#include <iostream>
#include "arrays.h"
#include "pair_interaction.h"
#include <math.h>
#include "Stokes.h"

using namespace arrays;

void moving_trap(Consts& consts, Coeffs& coeffs, TrapInfo& trapinfo, double& t){
    // Computes the position of the moving trap at time t
    // Changes: xpos0 (first three indidces are moving colloid)
    double time, deltat; 
    double &tstart = trapinfo.tstart;
    double &tstop = trapinfo.tstop;
    double &r12 = trapinfo.r12;
    double &twoPI = coeffs.twoPI;

    if ( (t - tstart) <0 ){
        time = trapinfo.tstart;
    }
    else if ( t >tstop ){
        time = trapinfo.tstop;
    }
    else{
        time = t;
    }
    deltat = time - tstart;
    // std::cout << "t: " << t << ", tstart: " << tstart << ", tstop: " << tstop << ", deltat: " << deltat << ", r12: " << r12 << std::endl;
    // xtrap[0] = r12 * cos(2.0*M_PI*deltat);
    // xtrap[1] = r12 * sin(2.0*M_PI*deltat);
    // xtrap[2] = 0.0;

    // Populate xpos with the position of the moving trap
    // xpos0[0] = r12 * cos(2.0*M_PI*deltat);
    // xpos0[1] = r12 * sin(2.0*M_PI*deltat);
    xpos0[0] = r12 * cos(twoPI*deltat);
    xpos0[1] = r12 * sin(twoPI*deltat);
    xpos0[2] = 0.0;

    // std:: cout << "Position of the moving trap: " << xpos0[0] << " " << xpos0[1] << " " << xpos0[2] << std::endl;
    return; 
}

void trapping_forces(PetscScalar fext[], Consts& consts, Coeffs& coeffs, ParticleInfo& pinfo, TrapInfo& trapinfo, double& t){
    PetscInt i, j, i3, ic3; 
    // Compute the trapping forces on the particles at time t
    moving_trap(consts, coeffs, trapinfo, t);
    // std::cout << "Position of the moving trap: " << xpos0[0] << " " << xpos0[1] << " " << xpos0[2] << std::endl;

    int& Np = pinfo.Np;
    int& Nm = pinfo.Nm;
    PetscInt& ndimp = consts.ndimp;
    double& ktrap = trapinfo.ktrap;

    if (t> trapinfo.tstop and trapinfo.weaken_trap >=0){
        ktrap = trapinfo.weaken_trap;
    }

    for( i = Nm; i < Np; i++){
        // std::cout << "in trapping forces i: " << i << std::endl;
        i3 = ndimp * i;         // global colloidal index
        ic3 = ndimp * (i - Nm); // local colloidal index

        // Print positions of moving particles
        // std::cout << "Position of moving particles: " << x[i3] << " " << x[i3+1] << " " << x[i3+2] << std::endl;

        // fext[ic3] = ktrap * ( x[i3] - xpos0[ic3] );
        // fext[ic3+1] = ktrap * ( x[i3+1] - xpos0[ic3+1] );
        // fext[ic3+2] = ktrap * ( x[i3+2] - xpos0[ic3+2] );

        fext[i3] += ktrap * ( x[i3] - xpos0[ic3] );
        fext[i3+1] += ktrap * ( x[i3+1] - xpos0[ic3+1] );
        fext[i3+2] += ktrap * ( x[i3+2] - xpos0[ic3+2] );

        // std::cout << "Trapping forces: " << fext[i3] << " " << fext[i3+1] << " " << fext[i3+2] << std::endl;

        // Compute trapping forces due to moving trap (index Np-2) or stationary traps
        // for (j = 0; j < ndimp; j++) {
        //     xdiff[j] = x[i3 + j] - (i == (Np - 2) ? xtrap[j] : xpos0[ic3 + j]);
        //     fext[i3+j] = ktrap * xdiff[j];
        // }
    }
    return; 
}

void harmonic_bond(PetscScalar *r, PetscScalar *f, Consts& consts, ParticleInfo& pinfo){
    // Computes the harmonic bond forces between particles
    for(int ii = 0; ii < consts.ndimp; ii++){
        f[ii] = -pinfo.kbond * (1 - (pinfo.r0/r[3])) * r[ii];
    }
    return;
}

void fene_bond(PetscScalar *r, PetscScalar *f, Consts& consts, ParticleInfo& pinfo){
    harmonic_bond(r,f, consts, pinfo);
    // f[0] = -pinfo.kbond * r[0];
    // f[1] = -pinfo.kbond * r[1];
    // f[2] = -pinfo.kbond * r[2];
    // double ext = r[3] / pinfo.Lmax;
    double ext2 = 1/(1- (r[3]/pinfo.Lmax)*(r[3]/pinfo.Lmax)); // stretching penalty 
    f[0] *= ext2;
    f[1] *= ext2;
    f[2] *= ext2;
    return;
}

void bond_forces(PetscScalar fext[], Consts& consts, ParticleInfo& pinfo, bool fene){
    int ii, jj, kk, ll, k3, j3; 
    PetscScalar f[consts.ndimp], r[consts.ndimp+1];

    for( ii = 0; ii < pinfo.nbonds; ii++){
        kk = bond_ids[0][ii];
        jj = bond_ids[1][ii];

        j3 = consts.ndimp * jj;
        k3 = consts.ndimp * kk;
        
        r[0] = x[k3] - x[j3];
        r[1] = x[k3 + 1] - x[j3 + 1];
        r[2] = x[k3 + 2] - x[j3 + 2];
        
        r[3] = sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] );

        if(fene){
            // std::cout << "using fene bond" << std::endl;
            fene_bond(r, f, consts, pinfo);
        }
        else{
            // std::cout << "using harmonic bond" << std::endl;
            harmonic_bond(r,f,consts,pinfo);
        }
        // std::cout << "Bond forces " << ii << " :" << f[0] << " " << f[1] << " " << f[2] << std::endl;

        for( ll = 0; ll < consts.ndimp; ll++){
            fext[k3 + ll] -= f[ll];
            fext[j3 + ll] += f[ll];
        }
    }

    return;
}

void PolyStokes::RHS(){
    // This routine constructs the right-hand side of the saddle point matrix
    
    // Zero out rhs
    // std::cout << "Constructing the right-hand side of the saddle point matrix" << std::endl;
    PetscInt& nm3 = consts.nm3;
    PetscInt& nc3 = consts.nc3;
    PetscInt& nm3nc3 = consts.nm3nc3;
    PetscInt& nm3nc11 = consts.nm3nc11;

    VecZeroEntries(rhs);
    PetscErrorCode ierr;
    PetscInt i, ishift, fintindices[nm3nc3];
    PetscScalar fint[nm3nc3];

    std::fill(fint, fint + nm3nc3, 0.0);
    
    pair_interaction(fint, consts, pinfo, dataStruct);
    trapping_forces(fint, consts, coeffs, pinfo, trapinfo, timeinfo.t);
    bond_forces(fint, consts, pinfo, fene);

    // Print fint
    // std::cout << "fint: " << std::endl;
    // for(i = 0; i < pinfo.Nc; i++){
    //     ishift = 3 * i + nm3; 
    //     std::cout << "fint[" << ishift << "]: " << fint[ishift] << std::endl;
    //     std::cout << "fint[" << ishift+1 << "]: " << fint[ishift+1] << std::endl;
    //     std::cout << "fint[" << ishift+2 << "]: " << fint[ishift+2] << std::endl;
    // }
    
    std::iota(fintindices, fintindices + nm3nc3, nm3nc11);
    // ierr = VecSetValues(rhs, nm3, fmindices, fbond, INSERT_VALUES); CHKERRV(ierr);
    // ierr = VecSetValues(rhs, nc3, fcindices, ftrap, INSERT_VALUES); CHKERRV(ierr);
    ierr = VecSetValues(rhs, nm3nc3, fintindices, fint, INSERT_VALUES); CHKERRV(ierr);

    // Print force values in rhs
    // PetscScalar *rhs_array;
    // VecGetArray(rhs, &rhs_array);
    // std::cout << "rhs: " << std::endl;
    // for(i = 0; i < pinfo.Nc; i++){
    //     ishift = 3 * i + nm3nc11 + nm3;
    //     std::cout << "rhs[" << ishift << "]: " << rhs_array[ishift] << std::endl;
    //     std::cout << "rhs[" << ishift+1 << "]: " << rhs_array[ishift+1] << std::endl;
    //     std::cout << "rhs[" << ishift+2 << "]: " << rhs_array[ishift+2] << std::endl;
    // }
    // VecRestoreArray(rhs, &rhs_array);

    // Assemble the rhs
    ierr = VecAssemblyBegin(rhs); CHKERRV(ierr);
    ierr = VecAssemblyEnd(rhs); CHKERRV(ierr);

    // Print the norm of the rhs
    // PetscScalar norm;
    // VecNorm(rhs, NORM_2, &norm);
    // PetscPrintf(PETSC_COMM_WORLD, "b norm: %g\n", norm);
    // VecView(rhs, PETSC_VIEWER_STDOUT_WORLD);

    // std::cout << "\n\n" << std::endl;

    return;
}