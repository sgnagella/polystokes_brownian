#include "Stokes.h"
#include <iostream>
#include "arrays.h"
#include <cmath>

using namespace arrays;

// Initializes the PolyStokes class and defines various constants and parameters
// for the calculations

PolyStokes::PolyStokes(double dt, int samplerate, double tmax, const std::string& output_dir, bool mm_HI, bool chain_HI, bool fene, bool record_forces, double t)
    : dt(dt), samplerate(samplerate), tmax(tmax), output_dir(output_dir), mm_HI(mm_HI), chain_HI(chain_HI), fene(fene), record_forces(record_forces), t(t)
{   
    std::cout << "using bond force fene: " << fene << std::endl;
    timeinfo.t = t;
    timeinfo.dt = dt;
    timeinfo.tmax = tmax;
    timeinfo.samplerate = samplerate;
    timeinfo.nsteps = floor(tmax / dt);
    init_coeffs();
}

PolyStokes::~PolyStokes()
{
    cleanup();
}

void PolyStokes::init_coeffs(){
    coeffs.twoPI = 2.0 * M_PI;
    coeffs.c1d2 = 1./2.;
    coeffs.c1d3 = 1./3.;
    coeffs.c2d3 = 2./3.;
    coeffs.c3d2 = 3./2.; 
    coeffs.c3d4 = 3./4.; 
    coeffs.c3d8 = 3./8.;
    coeffs.c4d3 = 4./3.;
    coeffs.c5d6 = 5./6.;
    coeffs.c9d4 = 9./4.; 
    coeffs.c9d5 = 9./5.;
    coeffs.c6d5 = 6./5.;
    coeffs.c9d8 = 9./8.; 
    coeffs.c9d2 = 9./2.;
    coeffs.c9d10 = 9./10.;
    coeffs.c9d20 = 9./20.;
    coeffs.c9d32 = 9./32.;
    coeffs.c18d5 = 18./5.;
    coeffs.c36d5 = 36./5.;
    coeffs.c54d5 = 54./5.;
}

void PolyStokes::set_consts(){
    consts.c50d49 = 50./49.;
    consts.c49d50 = 49./50.;
    consts.c50d49p49 = 2500. * std::pow(consts.c50d49, 49);
    consts.ndim = 3; 
    consts.const5 = 5;
    consts.const6 = 6;
    consts.n6 = 6 * pinfo.Np;
    consts.n5 = 5 * pinfo.Np;
    consts.n3 = 3 * pinfo.Np;
    consts.n4 = 4 * pinfo.Np;
    consts.nc4 = 4 * pinfo.Nc;
    consts.ndimp = consts.ndim;
    consts.n3p = consts.n3;
    consts.n6p = consts.n6;
    consts.n5p = consts.n5;
    consts.n11p = consts.n6p + consts.n5p;
    consts.n17p = consts.n11p + consts.n6p;
    consts.nm3 = 3 * pinfo.Nm;
    consts.nc3 = 3 * pinfo.Nc;
    consts.nc4 = 4 * pinfo.Nc;
    consts.nc5 = 5 * pinfo.Nc;
    consts.nc6 = 6 * pinfo.Nc;
    consts.nc11 = consts.nc5 + consts.nc6;
    consts.nm3nc3 = consts.nm3 + consts.nc3;
    consts.nm3nc6 = consts.nm3 + consts.nc6;
    consts.nm3nc11 = consts.nm3 + consts.nc11;
    consts.nm6nc17 = consts.nm3nc11 + consts.nm3nc6;
    consts.id_rows = 2;
    consts.pd_rows = 8;
    consts.PetscScalarSize = consts.nm3nc6 * sizeof(PetscScalar);

    int& ndim = consts.ndim;
    int& const5 = consts.const5;

    // Kronecker delta and Levi-Cevita
    // rank2_array delta(boost::extents[ndim][ndim]);
    // rank3_array eps(boost::extents[ndim][ndim][ndim]);

    // for (int i = 0; i < ndim; i++) {
    //     for (int j = 0; j < ndim; j++) {
    //         delta[i][j] = (i == j) ? 1.0 : 0.0;
    //     }
    // }
    
    // for (int i = 0; i < ndim; i++) {
    //     for (int j = 0; j < ndim; j++) {
    //         for (int k = 0; k < ndim; k++) {
    //             eps[i][j][k] = 0.0;
    //         }
    //     }
    // }

    // delta[0][0] = 1.; 
    // delta[1][1] = 1.;
    // delta[2][2] = 1.;

    // "Clockwise" progression of indices is positive 
    // eps[0][1][2] = 1.; 
    // eps[1][2][0] = 1.;
    // eps[2][0][1] = 1.;

    // // "Anti-clockwise" progression of indices is negative
    // eps[2][1][0] = -1; 
    // eps[1][0][2] = -1; 
    // eps[0][2][1] = -1;
    
    // per-particle mobility tensors 
    // Declare per-particle mobility tensors 
    // rank2_array mob_a(boost::extents[ndim][ndim]); // translation-force
    // rank2_array mob_b(boost::extents[ndim][ndim]); // rotation-force
    // rank2_array mob_bt(boost::extents[ndim][ndim]); // translation-torque ('TILDE')
    // rank2_array mob_c(boost::extents[ndim][ndim]); // rotation-torque
    // rank2_array mob_m(boost::extents[const5][const5]); // stress-strain
    // rank2_array mob_gt(boost::extents[ndim][const5]); 
    // rank2_array mob_ht(boost::extents[ndim][const5]);

    // rank3_array gt(boost::extents[ndim][ndim][ndim]); // translation-strain
    // rank3_array ht(boost::extents[ndim][ndim][ndim]); // rotation-strain
    // rank4_array m(boost::extents[ndim][ndim][ndim][ndim]); // stress-strain

    // std::cout << "Checking delta tensor:" << std::endl;
    // for (int i = 0; i < consts.ndim; i++) {
    //     for (int j = 0; j < consts.ndim; j++) {
    //         std::cout << delta[i][j] << " ";
    //         assert(!std::isnan(delta[i][j])); // Crash early if NaN
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Checking eps tensor:" << std::endl;
    // for (int i = 0; i < consts.ndim; i++) {
    //     for (int j = 0; j < consts.ndim; j++) {
    //         for (int k = 0; k < consts.ndim; k++) {
    //             std::cout << eps[i][j][k] << " ";
    //             assert(!std::isnan(eps[i][j][k])); // Crash early if NaN
    //         }
    //         std::cout << std::endl;
    //     }
    // }

}

void PolyStokes::particle_info(int Np, int Nc, int Nm, int Npoly, int Nmono_per_chain, double beta, double kbond, double r0, double Lmax, double tau)
{
    pinfo.Np = Np;
    pinfo.Nc = Nc;
    pinfo.Nm = Nm;
    pinfo.Npoly = Npoly;
    pinfo.Nmono_per_chain = Nmono_per_chain;
    pinfo.beta = beta;
    pinfo.kbond = kbond;
    pinfo.r0 = r0;
    pinfo.Lmax = Lmax;
    pinfo.tau = tau;

    // Print the input parameters
    std::cout << "Np: " << pinfo.Np << std::endl;
    std::cout << "Nc: " << pinfo.Nc << std::endl;
    std::cout << "Nm: " << pinfo.Nm << std::endl;
    std::cout << "Npoly: " << pinfo.Npoly << std::endl;
    std::cout << "Nmono_per_chain: " << pinfo.Nmono_per_chain << std::endl;
    std::cout << "beta: " << pinfo.beta << std::endl;
    std::cout << "kbond: " << pinfo.kbond << std::endl;
    std::cout << "r0: " << pinfo.r0 << std::endl;
    std::cout << "Lmax: " << pinfo.Lmax << std::endl;
    std::cout << "tau: " << pinfo.tau << std::endl;
    // std::cout << "fene" << pinfo.fene << std::endl;

    pinfo.npair = Np * (Np-1) / 2;
    pinfo.npair_AA = Nm * (Nm-1) / 2;
    pinfo.npair_BB = Nc * (Nc-1) / 2;
    pinfo.npair_AB = pinfo.npair - pinfo.npair_AA - pinfo.npair_BB;
    pinfo.nbonds_per_poly = Nmono_per_chain - 1;
    pinfo.nbonds = pinfo.nbonds_per_poly * Npoly;

    pinfo.beta_inv = 1.0 / beta;
    pinfo.beta2 = beta * beta;
    pinfo.beta3 = beta * beta * beta;

    set_consts();

    // Print delta and eps
    // std::cout << "delta tensor:" << std::endl;
    // for (int i = 0; i < consts.ndim; i++) {
    //     for (int j = 0; j < consts.ndim; j++) {
    //         std::cout << delta[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "eps tensor:" << std::endl;
    // for (int i = 0; i < consts.ndim; i++) {
    //     for (int j = 0; j < consts.ndim; j++) {
    //         for (int k = 0; k < consts.ndim; k++) {
    //             std::cout << eps[i][j][k] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }


}

void PolyStokes::trap_info(double ktrap, double tstart, double trun, double weaken_trap)
{
    trapinfo.ktrap = ktrap;
    trapinfo.tstart = tstart;
    trapinfo.trun = trun;
    trapinfo.tstop = tstart + trun;
    trapinfo.weaken_trap = weaken_trap;

    if( trapinfo.tstop > timeinfo.tmax ){
        std::cerr << "Simulation time exceeds maximum time" << std::endl;
    }
}

void PolyStokes::initial_configuration(pybind11::array_t<double> init_x0)
{
    int ii, i4, ishift;
    int& n3 = consts.n3;
    int& Nc = pinfo.Nc;
    PetscInt& nc3 = consts.nc3;
    PetscInt& nc4 = consts.nc4;
    PetscInt& nm3 = consts.nm3;
    auto buf = init_x0.request();
    double *ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.size; 

    std::cout << "initializing arrays" << std::endl;
    initialize_x(n3);
    initialize_xpos0(nc3);
    initialize_q(nc4);

    std::cout << "Copying initial positions" << std::endl;
    x.assign(ptr, ptr + n);

    // Store the initial positions of the trapped colloids

    std::cout << "Storing initial positions" << std::endl;
    for(ii = 0; ii < nc3; ii++){
        ishift = ii + nm3;
        xpos0[ii] = x[ishift];
    }

    std::cout << "Initial positions: " << xpos0[0] << " " << xpos0[1] << " " << xpos0[2] << std::endl;
    trapinfo.r12 = sqrt( xpos0[0]*xpos0[0] + xpos0[1]*xpos0[1] + xpos0[2]*xpos0[2] );
    std::cout << "Initial radial distance: " << trapinfo.r12 << std::endl;

    std::cout << "Initializing quaternion array" << std::endl;
    // initialize the quaternion array
    for(ii = 0; ii < Nc; ii++){
        i4 = 4 * ii;
        q[i4] = 1.0;
        q[i4 + 1] = 0.0;
        q[i4 + 2] = 0.0;
        q[i4 + 3] = 0.0;
    }

}

