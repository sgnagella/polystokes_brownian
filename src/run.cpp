// Main program to run conventional Stokesian Dynamics (unbounded)
// Dynamic time stepping of particle position subject to known forces and torques
// Project structure inspired by JWS SD Fortran code
// Author: Sachit G. Nagella 

// #include "run.h"
// #include "init.h"
// // #include "arrays.h"
// // #include "CellList.h"
// #include "pair_interaction.h"
// #include "globals.h"
// #include "check_dist.h"
// #include "mob.h"
// #include "lub.h"
// #include "traj.h"
// #include "saddle.h"
// #include "rhs.h"
// #include "vel.h"
// #include "write_configuration.h"
// #include "cleanup.h"
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <iostream>
// #include "config.h"

#include "Stokes.h"
#include "arrays.h"
#include <iostream>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <signal.h>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <petscsys.h>

// Needed for the predictor-corrector workspace (x, x_n, up, udiff, v_det_n/c, v_brown).
using namespace arrays;

namespace py = pybind11;

PolyStokes *sim_ptr = nullptr;  // Global pointer to simulation object

void PolyStokes::signal_handler(int signum) {
    std::cerr << "\nCaught signal " << signum << " (Ctrl+C). Cleaning up..." << std::endl;
    if (sim_ptr) {
        sim_ptr->cleanup();  // Call cleanup function
    }
    SlepcFinalize();  // Ensure SLEPc/PETSc finalization
    exit(signum);
}

// TODO: re use last solution as initial guess for the drift calcs
// Output the hydrodynamic forces on the colloids
void PolyStokes::run(){
    sim_ptr = this; 

    py::scoped_ostream_redirect stream;
    std::cout << "FTS Stokesian Dynamics (unbounded)" << std::endl;
        
    init();
    std::cout << "Initialization complete \n\n" << std::endl;
    // return;

    // Fill saddle point matrix with the self-interactions
    fill_self();

    // bool flag = true;

    for(int i = 0; i < timeinfo.nsteps; i++){
        timeinfo.t += timeinfo.dt;

        // Predictor-corrector (trapezoidal/Heun) time step. The stiff internal forces
        // (harmonic bonds, WCA) are integrated with an explicit-Euler predictor followed
        // by a corrector that re-evaluates the deterministic force at the predicted
        // position and averages -- this damps the overshoot that a plain Euler step can
        // produce for a stiff bond, which otherwise lets a monomer tunnel past the
        // colloid's WCA barrier in a single step. The Brownian slip velocity and the RFD
        // thermal drift are sampled ONCE at x_n (Euler for the stochastic part) and reused
        // unchanged for the corrector, preserving fluctuation-dissipation.
        //
        // x_{n+1} = x_n + 0.5*(v_det(x_n) + v_det(x_pred))*dt + v_brown(x_n)*dt
        //                + (kT/(2 eps)) * udiff(x_n) * dt
        x_n = x;   // snapshot the clean position before any perturbation this step

        // ---- Predictor stage: forces/mobility at x_n ----
        check_dist();
        if( !mm_HI && mono_ev ){
            build_monomer_cell_list();
        }
        RHS();          // rhs <- deterministic forces only (RHS() zeros rhs first)
        mob();          // A  <- mobility at x_n
        sync_mcc_schur_correction();   // Mcc_block <- eigenvalue-floored M^cc at x_n, before any solve reads it

        // Stage-2b milestone-1 check: with Mcm/Mcc populated, verify the distributed arrowhead
        // matvec against the serial one, then stop. Opt-in via POLYSTOKES_MPI_SELFTEST=1.
        if (i == 0 && std::getenv("POLYSTOKES_MPI_SELFTEST")) {
            bool ok_mv = verify_distributed_matvec();   // m1: operator
            bool ok_sv = verify_distributed_solve();    // m2: MINRES solve
            if (mpi_rank == 0)
                std::cout << "[mpi-selftest] matvec " << (ok_mv ? "PASS" : "FAIL")
                          << " | solve " << (ok_sv ? "PASS" : "FAIL") << std::endl;
            sim_ptr = nullptr;
            return;
        }

        solve_deterministic_vel(v_det_n);   // v_det_n = M(x_n) F(x_n)

        if (pinfo.kT > 0){
            solve_slip_vel();   // rhs += Brownian noise (added on top of the forces already there)
            new_vel();          // up <- v_det_n + v_brown_n (full predictor velocity)
            for(int k = 0; k < consts.nm3nc6; k++){ v_brown[k] = up[k] - v_det_n[k]; }
            drift();            // RFD thermal drift at x_n -> udiff (leaves x perturbed)
        }
        else{
            new_vel();          // up <- v_det_n (no noise)
            std::fill(v_brown.begin(), v_brown.end(), 0.0);
        }

        step();   // advance x from its drift()-perturbed state to x_pred, using `up`/`udiff`

        // ---- Corrector stage: forces/mobility at x_pred ----
        check_dist();
        if( !mm_HI && mono_ev ){
            build_monomer_cell_list();
        }
        RHS();
        mob();
        sync_mcc_schur_correction();   // Mcc_block <- eigenvalue-floored M^cc at x_pred

        solve_deterministic_vel(v_det_c);   // v_det_c = M(x_pred) F(x_pred)

        // ---- Final trapezoidal update from x_n ----
        for(int k = 0; k < consts.nm3nc6; k++){
            up[k] = 0.5 * (v_det_n[k] + v_det_c[k]) + v_brown[k];
        }
        step_from(x_n);   // x <- x_n + up*dt (+ drift term); no RFD probe to undo here
        step_quaternion();

        // if (flag){
        //     break;
        // }

        // Write the simulation ouptuts. Guard to rank 0: under MPI every rank currently holds
        // the full (replicated) state, so a single writer avoids all ranks clobbering the same
        // output files. Once state is distributed (Stage 2b/2c), the writers must first gather
        // the owned slices to rank 0.
        if (i == 0 or i % timeinfo.samplerate == 0){
            if (mpi_rank == 0){
                std::cout << "Time " << timeinfo.t << std::endl;
                write_configuration();
                write_quaternions();

                if(record_forces){
                    write_forces();
                }

                write_stresslets();
            }
        }
        // std::cout << "\n" << std::endl;
    }

    // [profiling] dump the PETSc/SLEPc -log_view summary without finalizing
    // (opt-in via POLYSTOKES_LOGVIEW=1; the default log handler is started in init()).
    if (std::getenv("POLYSTOKES_LOGVIEW")) { PetscLogView(PETSC_VIEWER_STDOUT_WORLD); }

    // Summary of negative-eigenvalue events collected over the run (always printed, whether or
    // not the per-event [schur] warnings were enabled).
    report_neg_eig_stats();

    sim_ptr = nullptr;

    return;
}


