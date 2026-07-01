#include <iostream>
#include "arrays.h"
#include <petscksp.h>
#include "Stokes.h"
using namespace arrays;

void PolyStokes::sample_slip_vel(){
    // This routine samples the random slip velocities 
    // needed to compute the far-field Brownian forces 
    int& nm3nc11 = consts.nm3nc11;
    for(int i = 0; i < nm3nc11; i++){ fb[i] = normal_dist(rng);}
    return;
}

void PolyStokes::solve_slip_vel(){
    // This routine computes the 11N Brownian slip velocities 
    // by computing the action of sqrt(M)*\Psi

    PetscErrorCode ierr; 
    VecZeroEntries(W);
    sample_slip_vel();

    PetscScalar *arr;
    ierr = VecGetArray(W, &arr); CHKERRV(ierr);
    std::copy(fb.begin(), fb.end(), arr);
    ierr = VecRestoreArray(W, &arr); CHKERRV(ierr);

    // Compute 
    ierr = MFNSolve(mfn, W, W); CHKERRV(ierr);

    // Check convergence
    MFNConvergedReason reason;
    MFNGetConvergedReason(mfn, &reason);
    if (reason < 0) {
        PetscPrintf(PETSC_COMM_WORLD, "MFN solver did not converge\n");
        return;
    }

    // Add to rhs
    PetscReal scale = PetscSqrtReal(2 * pinfo.kT / timeinfo.dt);
    VecScale(W, scale);

    // Create an index set defining where in the large vector to add W
    IS is;
    PetscInt start = 0;  // starting index in the large vector
    PetscInt size;
    VecGetLocalSize(W, &size);
    ISCreateStride(PETSC_COMM_WORLD, size, start, 1, &is);

    // Create scatter context
    VecScatter scatter;
    VecScatterCreate(W, NULL, rhs, is, &scatter);

    // Scatter with ADD_VALUES to add W into large_vec
    VecScatterBegin(scatter, W, rhs, ADD_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, W, rhs, ADD_VALUES, SCATTER_FORWARD);

    VecScatterDestroy(&scatter);
    ISDestroy(&is);

    return;
}