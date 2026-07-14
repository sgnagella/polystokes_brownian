#ifndef STOKES_H
#define STOKES_H
#include <vector>
#include <string>
#include <random>
#include <pybind11/numpy.h>
#include <petscksp.h>
#include <slepcmfn.h>
#include "params.h"
#include "data.h"
#include "box.h"
#include "multi_arrays.h"

class PolyStokes {
public:
    PolyStokes(double dt, int samplerate, double tmax, const std::string& output_dir, bool mm_HI=true, bool chain_HI=false, bool fene=true, bool record_forces=false, bool tether=false, bool mono_ev=false, const std::vector<double>& box=std::vector<double>(), double t=0.0);
    ~PolyStokes();
    void initial_configuration(pybind11::array_t<double> init_x0);
    void particle_info(double kT, double epsilon, int Np, int Nc, int Nm, int Npoly, int Nmono_per_chain, double beta, double kbond, double r0, double Lmax, double tau);
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
    bool tether; // whether to bond each chain's inner monomer to a host colloid (colloidal brush)
    bool mono_ev; // whether to apply monomer-monomer excluded-volume (WCA) interactions
    Box box;     // periodic box (inactive unless configured with 3 lengths)

    Consts consts;
    Coeffs coeffs;
    TrapInfo trapinfo;
    TimeInfo timeinfo;
    ParticleInfo pinfo; 
    Data dataStruct;

    KSP ksp;
    PC pc;
    MFN mfn = nullptr;   // full-M sqrt solver, created only when mm_HI (operator = A's mobility block)
    FN f = nullptr;

    // Dense workspace for the colloid Schur complement S = M^cc - beta*M^cm M^mc
    // (nc11 x nc11), used when mm_HI == false to sample Brownian slip velocities
    // via a block-Cholesky factor of the grand mobility. Its square root is taken
    // by an eigenvalue-floored symmetric eigendecomposition (see build_slip_vel_schur).
    Mat Smat = nullptr;

    bool petsc_finalized = false;
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist;

    // Monomer cell list, used only when (mm_HI == false && mono_ev) to find
    // monomer-monomer (AA) WCA neighbors in O(Nm) instead of the O(Nm^2) all-pairs
    // scan. Rebuilt each step by build_monomer_cell_list() and consumed by
    // monomer_wca(). Head-of-chain (cl_head, size nCells) + singly linked list
    // (cl_next, size Nm) over monomer indices 0..Nm-1.
    std::vector<int> cl_head;     // head monomer per cell (-1 = empty)
    std::vector<int> cl_next;     // next monomer in the same cell (-1 = end)
    int    cl_nc[3]   = {0, 0, 0};   // cells per axis
    double cl_size[3] = {0.0, 0.0, 0.0}; // cell edge per axis
    double cl_org[3]  = {0.0, 0.0, 0.0}; // lower corner (cell 0 origin) per axis
    bool   cl_periodic = false;   // apply periodic cell-index wrap in the neighbor scan

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
    void init_random(unsigned int seed);
    void init_square_root_solver();
    void init_solver();
    void check_dist();
    void build_monomer_cell_list();          // bin monomers into cl_head/cl_next (see cell_list.cpp)
    void monomer_wca(PetscScalar* fext);     // add monomer-monomer WCA via the cell list
    void RHS(bool drift=false);
    void mobility(double dr, double dr_inv, double dx, double dy, double dz, bool self, bool AB, bool AA);
    // Thread-safe monomer-colloid (AB) pair mobility: writes only to the passed buffers
    // (no global temporaries), so it can be called from an OpenMP-parallel AB loop.
    void mobility_AB(double dr, double dr_inv, double dx, double dy, double dz,
                     rank2_array& mob_a, rank2_array& mob_b,
                     rank2_array& mob_bt, rank2_array& mob_gt) const;
    void fill_self();
    void mob();
    void drift();
    void sample_drift_displacement();
    void solve_saddle(Vec X_out, bool warm_start=false);
    void sample_slip_vel();
    void solve_slip_vel();
    void build_slip_vel_schur();
    void schur_sqrt_lanczos(Mat Bcm, Mat Ccc, double beta, Vec b, Vec out, PetscInt kmax);
    void new_vel();
    void step();
    void step_quaternion();
    void write_configuration();
    void write_quaternions();
    void write_forces();
    void write_stresslets();
    void signal_handler(int signum);
    void cleanup();
};
#endif
