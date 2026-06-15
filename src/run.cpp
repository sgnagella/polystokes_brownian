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
#include <iostream>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <signal.h>
#include <atomic>
#include <petscsys.h> 

// using namespace arrays;

namespace py = pybind11;

PolyStokes *sim_ptr = nullptr;  // Global pointer to simulation object

void PolyStokes::signal_handler(int signum) {
    std::cerr << "\nCaught signal " << signum << " (Ctrl+C). Cleaning up..." << std::endl;
    if (sim_ptr) {
        sim_ptr->cleanup();  // Call cleanup function
    }
    PetscFinalize();  // Ensure PETSc finalization
    exit(signum);
}

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
        // std::cout << "Time step: " << i << std::endl;
        // std::cout << "Time: " << t << std::endl;
        // Compute pairwise distances after reading initial particle config
        // std::cout << "Checking pairwise distances..." << std::endl;
        check_dist();

        // Construct RHS and correct for any overlaps
        // std::cout << "Computing RHS" << std::endl;
        RHS();
        
        // Compute the cell list
        // std::cout << "Computing the cell list..." << std::endl;
        // compute_cell_list();

        // std::cout << "Computing the mobility matrix..." << std::endl;
        // Compute the grand mobility matrix and insert to saddle point matrix
        mob();
        
        // Construct the saddle point matrix
        // saddle();

        // Add lubrication
        // add_lub();

        // Get new particle velocities 
        new_vel();
        
        // std::cout << "\n" << std::endl;

        // Time advance the particle positions
        step();
        step_quaternion();

        // if (flag){
        //     break;
        // }

        // Write the simulation ouptuts
        // write_outputs();
        if (i == 0 or i % timeinfo.samplerate == 0){
            std::cout << "Time " << timeinfo.t << std::endl;
            write_configuration();
            write_quaternions();

            if(record_forces){
                write_forces();
            }
        }
        // std::cout << "\n" << std::endl;
    }

    sim_ptr = nullptr;

    return;
}


