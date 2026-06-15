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

    extern std::vector<double> fext; 
    extern std::vector<double> uext;

    extern std::vector<double> uinf;
    extern std::vector<double> einf;
    extern std::vector<double> sh;

    extern std::vector<double> fb;
    extern std::vector<double> sb;

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

    extern Vec X;
    extern Vec rhs;

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

    // void initialize_M(PetscInt nm3nc11);

    void initialize_B(PetscInt nm3nc11, PetscInt nm3nc6);

    void initialize_A(PetscInt nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6);

    void initialize_X(PetscInt nm6nc17);

    void initialize_rhs(PetscInt nm6nc17);                  
};

#endif