#include <iostream>
#include "arrays.h"
#include <petscksp.h>
#include "Stokes.h"
using namespace arrays;

void PolyStokes::solve_saddle(Vec X_out, bool warm_start){
    // This routine solves the saddle point system
    // using a Krylov subspace method.
    // warm_start: if true, use the current contents of X_out as the initial
    // guess; if false, KSP zeroes X_out and starts cold.

    // Distributed (MPI) path: on >1 rank the serial KSP/operator/vectors are replicated-SELF
    // and unused; the solve is done by the monomer-partitioned MINRES solver instead. It reads
    // the replicated rhs and writes the full solution back into X_out (also SELF/replicated).
    if (mpi_size > 1) { solve_saddle_distributed(X_out, warm_start); return; }


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

    ierr = KSPSetInitialGuessNonzero(ksp, warm_start ? PETSC_TRUE : PETSC_FALSE); CHKERRV(ierr);
    ierr = KSPSolve(ksp, rhs, X_out); CHKERRV(ierr);
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
    // Cold start: this solve includes the fresh Brownian slip velocity (rhs was augmented
    // by solve_slip_vel()), so the previous step's solution is uncorrelated with this one
    // and warm-starting it measurably INCREASES iterations. Keep it cold.
    solve_saddle(X, false);

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

void PolyStokes::solve_deterministic_vel(std::vector<double>& out){
    // Deterministic-only velocity solve for the predictor-corrector scheme: solves the
    // saddle system with the current `rhs` (holding just the internal forces from
    // RHS()+mob(), no Brownian noise) into the dedicated Xdet vector, so it never
    // clobbers the full solve's X (or fext, unlike new_vel() with record_forces). Used
    // once at x_n (predictor) and once at x_pred (corrector); see run.cpp.
    // Cold start. Warm-starting from Xdet's previous contents was measured to INCREASE
    // GMRES iterations (~19.5 -> ~23.7 matvecs/solve) even for the well-correlated corrector:
    // with PETSc's ||b||-relative convergence, the reassembled operator + adaptive Schur
    // floor make the prior solution no better than a zero guess, so it only adds iterations.
    // See docs/profiling_baseline_serial.md.
    solve_saddle(Xdet, false);
    const PetscScalar *X_array;
    VecGetArrayRead(Xdet, &X_array);
    memcpy(out.data(), &X_array[consts.nm3nc11], consts.PetscScalarSize);
    VecRestoreArrayRead(Xdet, &X_array);
    return;
}