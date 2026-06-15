#ifndef PARAMS_H
#define PARAMS_H

#include <petscksp.h>

// Define struct containing polymer parameters
struct ParticleInfo{
    int Np; 
    int Nmono_per_chain;
    int Nc;
    int Nm;
    double beta;
    int npair;
    int npair_AA;
    int npair_AB;
    int npair_BB;
    int Npoly;
    int nbonds;
    int nbonds_per_poly;
    double beta_inv;
    double beta2;
    double beta3;
    double r0; 
    double kbond;
    double Lmax; 
    double tau;
    bool fene;
}; 

struct Consts{
    double c50d49;
    double c49d50;
    double c50d49p49;
    int ndim;
    int const5;
    int const6;
    int n6; 
    int n5; 
    int n3; 
    int n4;
    int id_rows;
    int pd_rows;
    PetscInt ndimp; 
    PetscInt n3p;
    PetscInt n6p; 
    PetscInt n5p;
    PetscInt n11p;
    PetscInt n17p;
    PetscInt nm3;
    PetscInt nc3;
    PetscInt nc4; 
    PetscInt nc5;
    PetscInt nc6;
    PetscInt nc11; 
    PetscInt nm3nc3; 
    PetscInt nm3nc6; 
    PetscInt nm3nc11;
    PetscInt nm6nc17;
    std::size_t PetscScalarSize;
};

struct TrapInfo{
    PetscScalar ktrap;
    PetscScalar r12;
    PetscScalar tstart; 
    PetscScalar trun;
    PetscScalar tstop;
    PetscScalar weaken_trap;
}; 

struct TimeInfo{
    PetscScalar t;
    PetscInt nsteps;
    PetscScalar dt;
    PetscInt samplerate;
    PetscScalar tmax;
};

// Declare coefficients of the inverse power series for mobility funcs
struct Coeffs{
    double twoPI;
    double c1d2;
    double c1d3;
    double c2d3;
    double c3d2;
    double c3d4;
    double c3d8;
    double c4d3;
    double c5d6;
    double c9d4;
    double c9d5;
    double c6d5;
    double c9d8;
    double c9d2;
    double c9d10;
    double c9d20;
    double c9d32;
    double c18d5;
    double c36d5;
    double c54d5;
};


#endif