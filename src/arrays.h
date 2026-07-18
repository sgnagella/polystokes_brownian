// Header file to declare multidimensional arrays

#ifndef ARRAY_H
#define ARRAY_H

#include "multi_arrays.h"
#include <petscmat.h>

namespace arrays{
    // Declare arrays of doubles for position data
    extern std::vector<double> radii; 
    extern std::vector<double> x; 
    extern std::vector<double> xpos0;
    extern std::vector<double> x0; 
    extern std::vector<double> q;

    // Declare arrays of doubles for velocity data
    extern std::vector<double> u;
    extern std::vector<double> up;
    extern std::vector<double> udiff;

    // Predictor-corrector (trapezoidal/Heun) workspace, used to damp explicit-step
    // overshoot of stiff forces (bonds, WCA) that could otherwise push a monomer
    // through the colloid. x_n snapshots the position at the start of the step (size
    // n3); v_det_n/v_det_c hold the deterministic (force-only) velocity at x_n and at
    // the predicted position x_pred (size nm3nc6); v_brown holds the Brownian slip
    // velocity, sampled once at x_n and reused unchanged for the corrector (Euler for
    // the stochastic part, Heun for the deterministic part). See run.cpp.
    extern std::vector<double> x_n;
    extern std::vector<double> v_det_n;
    extern std::vector<double> v_det_c;
    extern std::vector<double> v_brown;

    extern std::vector<double> fext; 
    extern std::vector<double> uext;

    extern std::vector<double> uinf;
    extern std::vector<double> einf;
    extern std::vector<double> sh;

    extern std::vector<double> fb;
    extern std::vector<double> sb;

    // Cached Schur eigendecomposition of the colloid Schur complement
    // S = M^cc - beta*M^cm(M^cm)^T (dense path, i.e. nc11 below the Lanczos crossover).
    // Refreshed once per stage by sync_mcc_schur_correction() -- called right after
    // mob(), before any solve reads Mcc_block -- so that solve_deterministic_vel() and
    // the Brownian noise sample in build_slip_vel_schur() agree on M^cc within a stage.
    // schur_Q is nc11 x nc11, column-major (schur_Q[i + j*nc11] = Q[i,j]);
    // schur_lambda_corrected[j] = max(lambda_j, eps) is the eigenvalue-floored spectrum.
    extern std::vector<PetscScalar> schur_Q;
    extern std::vector<PetscReal>   schur_lambda_corrected;

    extern std::vector<double> addrn;
    extern std::vector<double> drift;

    // Use boost libraries to create multidimensional arrays     
    extern std::vector<std::vector<long double>> pd; 
    extern std::vector<std::vector<int>> id;
    extern std::vector<std::vector<int>> bond_ids;
    extern std::vector<int> chain_ids;
    extern std::vector<std::vector<int>> vlist;
    extern std::vector<std::vector<int>> bond_list;
    extern std::vector<int> pair_types;
    extern std::vector<std::vector<int>> id_AA; 
    extern std::vector<int> id_AB;
    extern std::vector<int> id_BB;
    extern std::vector<std::vector<int>> mesid; 

    extern rank2_array uold;

    extern rank2_array rfu; 
    extern rank2_array zmuf;
    extern rank2_array rfe; 
    extern rank2_array zmus;
    extern rank2_array rsu;
    extern rank2_array rse;
    extern rank2_array zmes;
    // extern rank2_array pmob;

    // extern rank2_array rfu11;
    // extern rank2_array rfu22; 
    // extern rank2_array rfu12;
    // extern rank2_array rfu21;
    
    extern rank2_array rtfu;
    extern rank2_array rfu_s;

    // extern Mat ZMUF;
    // extern Mat ZMUS;
    // extern Mat ZMES;
    extern Mat Pinv;
    extern Mat M;
    extern Mat B;
    extern Mat A;

    // Arrowhead mobility pieces for the single-colloid / no-monomer-HI regime:
    // M^mm = beta_inv*I (diagonal, not stored), M^cm = colloid<->monomer coupling
    // (nc11 x nm3, rebuilt each step, O(N)), M^cc = colloid self-mobility (nc11 x nc11,
    // constant). These are the mobility representation shared by the (matrix-free) saddle
    // operator and the Brownian slip square root, so neither needs a dense grand mobility.
    extern Mat Mcm_block;
    extern Mat Mcc_block;
    // Pristine, never-modified colloid self-mobility (raw physics, set once in
    // fill_self()). Mcc_block is reset from this each step and given an adaptive,
    // per-eigen-direction correction (build_slip_vel_schur) so the correction never
    // accumulates across steps.
    extern Mat Mcc_base;

    extern Vec X;
    extern Vec Xd;
    extern Vec Xdet;   // predictor-corrector deterministic-only solve (see solve_deterministic_vel)
    extern Vec rhs;
    extern Vec W;

    // Communicator for the replicated per-step vectors (rhs/W/X/Xd/Xdet). PETSC_COMM_WORLD in
    // serial; set to PETSC_COMM_SELF in init() when running on >1 MPI rank, so each rank holds a
    // full replicated copy (assembly stays replicated; only the distributed solve is collective).
    // Must be assigned before alok_arrays().
    extern MPI_Comm rep_comm;

    // extern rank2_array zm; // grand mobility matrix

    // Functions to initialize the global arrays
    void initialize_radii(int size);
    void initialize_x(int size);
    void initialize_xpos0(int size);
    void initialize_x0(int size);
    void initialize_up(int size);
    void initialize_q(int size);

    void initialize_u(int size);
    void initialize_udiff(int size);

    void initialize_x_n(int size);
    void initialize_v_det_n(int size);
    void initialize_v_det_c(int size);
    void initialize_v_brown(int size);

    void initialize_uinf(int size);
    void initialize_einf(int size);
    void initialize_sh(int size);

    void initialize_fb(int size);
    void initialize_sb(int size);

    void initialize_addrn(int size);
    void initialize_drift(int size);

    void initialize_fext(int size);
    void initialize_uext(int size);

    void initialize_pd(int rows, int cols);
    void initialize_bondid(int rows, int cols);
    void initialize_chainid(int size);
    void initialize_id(int rows, int cols);
    void initialize_vlist(int rows);
    void initialize_bond_list(int rows, int cols);
    void initialize_pair_types(int size);
    void initialize_id_AA(int rows, int cols); 
    void initialize_id_AB(int size); 
    void initialize_id_BB(int size);    
    void initialize_mesid(int rows, int cols);

    // void initialize_rfu(int rows, int cols);
    void initialize_uold();

    void initialize_rfu(); 
    void initialize_zmuf();
    void initialize_rfe();
    void initialize_zmus();
    void initialize_rsu();
    void initialize_rse();
    void initialize_zmes();


    // void initialize_zm();

    // void initialize_ZMUF();

    // void initialize_ZMUS();

    // void initialize_ZMES();

    // void initialize_Pinv(Petscint nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6);

    void initialize_M(PetscInt nm3nc11);
    void initialize_arrowhead(PetscInt nc11, PetscInt nm3);

    void initialize_B(PetscInt nm3nc11, PetscInt nm3nc6);

    void initialize_A(PetscInt nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6);
    void initialize_A_shell(PetscInt nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6,
                            PetscInt nm3, PetscInt nc11, PetscInt Nc, PetscReal beta_inv);

    void initialize_X(PetscInt nm6nc17);

    void initialize_Xd(PetscInt nm6nc17);

    void initialize_Xdet(PetscInt nm6nc17);

    void initialize_W(PetscInt nm3nc11);

    void initialize_rhs(PetscInt nm6nc17);                  
};

#endif