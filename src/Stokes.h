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

// Lightweight RAII wrapper around a PETSc log event: begins the event on
// construction and ends it when the scope exits (any return path). Used to time
// the plain-C++ assembly routines (mob/check_dist/RHS/drift) -- which are otherwise
// invisible to -log_view -- so the Stage-0 profiling baseline and the later threaded
// comparison get a per-function wall-time split. A no-op cost when logging is off.
struct PetscEventScope {
    PetscLogEvent ev;
    explicit PetscEventScope(PetscLogEvent e) : ev(e) { PetscLogEventBegin(ev, 0, 0, 0, 0); }
    ~PetscEventScope() { PetscLogEventEnd(ev, 0, 0, 0, 0); }
};

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

    // MPI rank/size on PETSC_COMM_WORLD (set in init()). Default 0/1 so the serial path is
    // unchanged. Stage-2 uses these to guard I/O to rank 0 and (later) to partition monomers.
    PetscMPIInt mpi_rank = 0, mpi_size = 1;

    // PETSc log-event handles for the O(Nm) assembly routines (registered in init()).
    // Let -log_view report mob/check_dist/RHS/drift as named events; KSPSolve and the
    // arrowhead MatMult are already logged automatically by PETSc.
    PetscLogEvent ev_mob = 0, ev_check = 0, ev_rhs = 0, ev_drift = 0;

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
    // Stage-2b verification: apply the arrowhead operator to a deterministic vector both
    // serially (replicated) and via the distributed monomer-partition + colloid MPI_Allreduce,
    // and report the max discrepancy. Env-triggered (POLYSTOKES_MPI_SELFTEST) from run().
    bool verify_distributed_matvec();
    // Stage-2b m2: solve the CURRENT saddle system both serially and via a distributed
    // MINRES over the monomer-partitioned operator, and report agreement. Env-triggered.
    bool verify_distributed_solve();
    void mob();
    void drift();
    void sample_drift_displacement();
    void solve_saddle(Vec X_out, bool warm_start=false);
    void sample_slip_vel();
    void solve_slip_vel();
    // Refresh Mcc_block = Mcc_base + adaptive eigenvalue-floor correction of the colloid
    // Schur complement, using the CURRENT (this-stage) Mcm_block. Must be called once per
    // stage, right after mob() and before any solve reads Mcc_block, so the deterministic
    // solve and the Brownian noise sample agree on M^cc -- see run.cpp and slip_vel.cpp.
    void sync_mcc_schur_correction();
    void build_slip_vel_schur();
    void schur_sqrt_lanczos(Mat Bcm, Mat Ccc, double beta, Vec b, Vec out, PetscInt kmax);
    void new_vel();
    // Solve the saddle system with the CURRENT rhs/A (assumed set up by a prior
    // RHS()+mob() call, with rhs holding forces only -- i.e. called before
    // solve_slip_vel() adds Brownian noise) and extract the deterministic velocity
    // block (size nm3nc6) into `out`. Used by the predictor-corrector step in run().
    void solve_deterministic_vel(std::vector<double>& out);
    void step();
    // Advance positions from a saved, unperturbed base position x_base using the
    // current `up` (velocity) and `udiff` (RFD drift estimate, already computed at
    // x_base) -- used for the predictor-corrector's final corrector update, where
    // (unlike step()) there is no RFD probe to undo since x_base is already clean.
    void step_from(const std::vector<double>& x_base);
    void step_quaternion();
    void write_configuration();
    void write_quaternions();
    void write_forces();
    void write_stresslets();
    void signal_handler(int signum);
    void cleanup();
};
#endif
