#include "arrays.h"
#include "quat.h"
#include "Stokes.h"
#include <iostream> 
#include <cmath>

using namespace arrays;

void PolyStokes::step(){
    // Time-advance the particle positions

    // std::cout << "Time advancing the trajectory..." << std::endl;
    for( int i = 0; i < consts.n3; i++){
        x[i] += up[i]*timeinfo.dt;
    }
    return;
}

void PolyStokes::step_quaternion(){
    // Time-advance the particle orientations

    int i, i3, i4;
    int& Nc = pinfo.Nc;
    PetscInt& nm3nc3 = consts.nm3nc3;
    double& dt = timeinfo.dt;

    // std::cout << "Time advancing the orientations..." << std::endl;
    std::vector<double> qi(4), omega(consts.ndim);

    for(i = 0; i < Nc; i++){
        i4 = 4 * i;
        i3 = 3 * i + nm3nc3;
        qi[0] = q[i4];
        qi[1] = q[i4+1];
        qi[2] = q[i4+2];
        qi[3] = q[i4+3];
        // std::cout << "up[i3]: " << up[i3] << "up[i3+1] " << up[i3+1] << "up[i3+2] " << up[i3+2] << std::endl;
        omega[0] = up[i3];
        omega[1] = up[i3 + 1];
        omega[2] = up[i3 + 2];
        // std::cout << "Omega: " << omega[0] << " " << omega[1] << " " << omega[2] << std::endl;
        qi = integrate_quat(qi, omega, dt);
        q[i4] = qi[0];
        q[i4+1] = qi[1];
        q[i4+2] = qi[2];
        q[i4+3] = qi[3];
    }
    return;
}