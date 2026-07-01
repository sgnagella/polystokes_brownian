#include "arrays.h"
#include "quat.h"
#include "Stokes.h"
#include <iostream> 
#include <cmath>

using namespace arrays;

void PolyStokes::step(){
    // Time-advance the particle positions

    if(pinfo.kT >0){
        // Thermal drift kT*div(M)*dt from the RFD estimate in drift(). The two mobility
        // probes are separated by 2*rfd_eps, so div(M) ~ udiff/(2*rfd_eps); the drift
        // displacement is therefore (kT/(2*rfd_eps)) * udiff * dt.
        double drift_scale = pinfo.kT / (2.0 * consts.rfd_eps);
        for (int i = 0; i < consts.n3; i++){
            x[i] += consts.rfd_eps * fext[i]; // undo RFD probe: drift() left positions at x - rfd_eps*W
            x[i] += drift_scale * udiff[i] * timeinfo.dt;
        }
    }

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