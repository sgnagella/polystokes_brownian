#ifndef STOKES_H
#define STOKES_H
#include <vector>
#include <string>
#include <pybind11/numpy.h>
#include <petscksp.h>
#include "params.h"
#include "data.h"
#include "multi_arrays.h"

class PolyStokes {
public:
    PolyStokes(double dt, int samplerate, double tmax, const std::string& output_dir, bool mm_HI=true, bool chain_HI=false, bool fene=true, bool record_forces=false, double t=0.0);
    ~PolyStokes();
    void initial_configuration(pybind11::array_t<double> init_x0);
    void particle_info(int Np, int Nc, int Nm, int Npoly, int Nmono_per_chain, double beta, double kbond, double r0, double Lmax, double tau);
    void trap_info(double ktrap, double tstart, double trun, double weaken_trap=-1);
    void run();

private:
    double t;
    double dt;
    int samplerate;
    double tmax;
    std::string output_dir;
    bool mm_HI; // whether to include monomer-monomer HI
    bool chain_HI; // only consider HI between monomers of differing chains (mm_HI needs to be true)
    bool fene;
    bool record_forces;

    Consts consts;
    Coeffs coeffs;
    TrapInfo trapinfo;
    TimeInfo timeinfo;
    ParticleInfo pinfo; 
    Data dataStruct;

    KSP ksp;
    PC pc;
    bool petsc_finalized = false;

    // rank2_array delta;
    // rank3_array eps;
    // // Declare per-particle mobility tensors 
    // rank2_array mob_a; // translation-force
    // rank2_array mob_b; // rotation-force
    // rank2_array mob_bt; // translation-torque ('TILDE')
    // rank2_array mob_c; // rotation-torque
    // rank2_array mob_m; // stress-strain
    // rank2_array mob_gt; 
    // rank2_array mob_ht;

    // rank3_array gt; // translation-strain
    // rank3_array ht; // rotation-strain
    // rank4_array m; // stress-strain
    
    void init_coeffs();
    void set_consts();
    void init();
    void init_solver();
    void check_dist();
    void RHS();
    void mobility(double dr, double dr_inv, double dx, double dy, double dz, bool self, bool AB, bool AA);
    void fill_self();
    void mob();
    void solve_saddle();
    void new_vel();
    void step();
    void step_quaternion();
    void write_configuration();
    void write_quaternions();
    void write_forces();
    void signal_handler(int signum);
    void cleanup();
};
#endif
