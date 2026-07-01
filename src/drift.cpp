#include "Stokes.h"
#include "arrays.h"
#include "multi_arrays.h"
#include <iostream>
#include <cmath>
#include <random>
using namespace arrays;

void PolyStokes::sample_drift_displacement(){
    // This routine samples the drift displacement for the Brownian dynamics update
    // It uses the fluctuation-dissipation relation to sample the drift from the mobility matrix
    int& nm3nc6 = consts.nm3nc6;
    for(int i = 0; i < nm3nc6; i++){ fext[i] = normal_dist(rng);}
    return;
}

void PolyStokes::drift(){
    int& nm3nc6 = consts.nm3nc6;
    // This routine computes the drift term in the Brownian dynamics update
    sample_drift_displacement();

    // Apply positive random displacements (positions only, like traj.cpp)
    for( int i = 0; i < consts.n3; i++){ x[i] += 0.001 * fext[i]; }
    check_dist();
    mob();
    RHS(true);

    // Solve for the drift velocities
    // 1st drift solve: cold start (drift RHS is freshly random each step)
    solve_saddle(Xd, false);
    const PetscScalar *X_array;
    VecGetArrayRead(Xd, &X_array);
    memcpy(arrays::drift.data(), &X_array[consts.nm3nc11], consts.PetscScalarSize);
    VecRestoreArrayRead(Xd, &X_array);

    // Apply negative random displacements (positions only, like traj.cpp)
    for( int i = 0; i < consts.n3; i++){ x[i] -= 0.002 * fext[i]; }
    check_dist();
    mob();
    RHS(true);

    // Solve for the drift velocities
    // 2nd drift solve: warm-start from the 1st drift solve (same RHS, tiny position change)
    solve_saddle(Xd, true);
    VecGetArrayRead(Xd, &X_array);
    for(int i = 0; i < nm3nc6; i++){ udiff[i] = X_array[consts.nm3nc11 + i] - arrays::drift[i]; }
    VecRestoreArrayRead(Xd, &X_array);

    return;
}