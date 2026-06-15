#include <iostream>
#include "arrays.h"
#include <petscksp.h>
#include "Stokes.h"
using namespace arrays;

void PolyStokes::solve_saddle(){
    // This routine solves the saddle point system
    // using a Krylov subspace method

    // Declare the KSP solver
    PetscErrorCode ierr;
    KSPConvergedReason reason;
    PetscReal norm;
    PetscInt its;
    int ishift;
    // KSPCreate(PETSC_COMM_WORLD, &ksp);

    // ierr = KSPSetOperators(ksp, A, A); CHKERRV(ierr);
    // std::cout << "In solve_saddle at time " << timeinfo.t << std::endl;
    // Print force values in rhs
    // PetscScalar *rhs_array;
    // VecGetArray(rhs, &rhs_array);
    // std::cout << "rhs: " << std::endl;
    // for(int i = 0; i < pinfo.Nc; i++){
    //     ishift = 3 * i + consts.nm3nc11 + consts.nm3;
    //     std::cout << "rhs[" << ishift << "]: " << rhs_array[ishift] << std::endl;
    //     std::cout << "rhs[" << ishift+1 << "]: " << rhs_array[ishift+1] << std::endl;
    //     std::cout << "rhs[" << ishift+2 << "]: " << rhs_array[ishift+2] << std::endl;
    // }
    // VecRestoreArray(rhs, &rhs_array);

    ierr = KSPSolve(ksp, rhs, X); CHKERRV(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRV(ierr);

    if (reason < 0) {
        KSPGetIterationNumber(ksp, &its);
        PetscPrintf(PETSC_COMM_WORLD, "Iterations: %D\n", its);
        ierr = KSPGetResidualNorm(ksp, &norm); CHKERRV(ierr);
        std::cout << "Norm of the error: " << norm << std::endl;
        std::cout << "KSPSolve failed to converge" << std::endl;
    }

    // Print the colloidal velocities 
    // PetscScalar *X_array;
    // VecGetArray(X, &X_array);
    // std::cout << "X: " << std::endl;
    // for(int i = 0; i < pinfo.Nc; i++){
    //     ishift = 3 * i + consts.nm3nc11 + consts.nm3;
    //     std::cout << "X[" << ishift << "]: " << X_array[ishift] << std::endl;
    //     std::cout << "X[" << ishift+1 << "]: " << X_array[ishift+1] << std::endl;
    //     std::cout << "X[" << ishift+2 << "]: " << X_array[ishift+2] << std::endl;
    // }
    // VecRestoreArray(X, &X_array);

    return;
}

void PolyStokes::new_vel(){
    // This routine computes the new velocities of the particles
    // by solving the saddle point system

    PetscInt i, ishift;
    solve_saddle();

    // Get the translational and rotational 
    // velocities from the solution vector 
    const PetscScalar *X_array;
    VecGetArrayRead(X, &X_array);
    memcpy(up.data(), &X_array[consts.nm3nc11], consts.PetscScalarSize);
    if(record_forces){
        memcpy(fext.data(), X_array, consts.nm3nc6 * sizeof(PetscScalar));
    }
    // Print the elements of up
    // std::cout << "up: " << std::endl;
    // for(i = 0; i < consts.nm3nc6; i++){
    //     std::cout << up[i] << std::endl;
    // }
    VecRestoreArrayRead(X, &X_array);

    return;
}