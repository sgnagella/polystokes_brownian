// Header files for namespace declaration 

#ifndef GLOBALS_H // Check if GLOBALS_H is not defined
#define GLOBALS_H // If not defined, define it

#include "arrays.h"
#include <iostream>
#include <petscksp.h>

// Delcare the solver
extern KSP ksp;
extern PC pc;

// To be defined in init.cpp
extern int ndim;
extern int const5;
extern int const6;
// extern int Nc;
// extern int Nmono_per_chain;
extern int Nm;
extern int Np; 
// extern int Npoly;
extern int npair;
extern int npair_AA;
extern int npair_AB;
extern int npair_BB;
extern int nbonds_per_poly;
extern int nbonds;
extern int pd_rows; 
extern int id_rows; 
extern int n11;
extern int n6; 
extern int n5; 
extern int n3;
extern int n4;
extern int nc4;

extern PetscInt ndimp;
extern PetscInt n3p;
extern PetscInt n6p; 
extern PetscInt n5p;
extern PetscInt n11p;
extern PetscInt n17p;
extern PetscInt nm3;
extern PetscInt nc3;
extern PetscInt nc5;
extern PetscInt nc6;
extern PetscInt nc11; 
extern PetscInt nm3nc3; 
extern PetscInt nm3nc6; 
extern PetscInt nm3nc11; 
extern PetscInt nm6nc17; 

// Simulation parameters
extern PetscScalar t;
// extern PetscInt samplerate;
// extern PetscScalar dt; 
extern PetscScalar ktrap;
extern PetscScalar kbond;
extern PetscScalar r0; 
extern PetscScalar Lmax;
extern PetscScalar tstart; 
extern PetscScalar trun;
extern PetscScalar tstop;
extern PetscInt nsteps; 
extern PetscScalar r12;

// Declare coefficients of the inverse power series for mobility funcs
extern double twoPI;
extern double c1d2;
extern double c1d3;
extern double c2d3;
extern double c3d2;
extern double c3d4;
extern double c3d8;
extern double c4d3;
extern double c5d6;
extern double c9d4;
extern double c9d5;
extern double c6d5;
extern double c9d8;
extern double c9d2;
extern double c9d10;
extern double c9d20;
extern double c18d5;
extern double c36d5;
extern double c54d5;

extern double beta; 
extern double beta_inv;
extern double beta2; 
extern double beta3;

extern double tau;

extern std::size_t PetscScalarSize;

// Declare coefficients of the inverse power series for resistance funcs
// X11a
extern double cX11a4;
extern double cX11a6;
extern double cX11a8;

// X12a
extern double cX12a3;
extern double cX12a5;
extern double cX12a7;
extern double cX12a9;

// Y11a
extern double cY11a2;
extern double cY11a4;
extern double cY11a6;
extern double cY11a8;

// Y12a
extern double cY12a3;
extern double cY12a5;
extern double cY12a7;
extern double cY12a9;

// Y11b
// double cY11b3 = 0.75 ;
extern double cY11b5;
extern double cY11b7;
extern double cY11b9;

// Y12b
// For now, use the coeffs as found in RJP thesis appendix
// double cY12b2 = 1.;
extern double cY12b4;
extern double cY12b6;
extern double cY12b8;

// Y11c
// double cY11c4 = 1.;
extern double cY11c6;
extern double cY11c8;

// Y12c
extern double cY12c7;
extern double cY12c9;

// Misc. constants
extern double one6;
extern double one15;
extern double c3d112;
extern double one12;
extern double ug1;
extern double ug2;
extern double twopi; 

// Kronecker delta
extern rank2_array delta;

// Levi Cevita tensor
extern rank3_array eps;

// Files for writing outputs
// std::ofstream output_file1; 
// std::ofstream output_file2;

#endif // End of conditional check 