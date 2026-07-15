// This programs defines the namespace and contains function definition for 
// creating 2d arrays for the conventional SD calculation

#include "multi_arrays.h"
#include "arrays.h"
#include <iostream>

namespace arrays{
    // Declare arrays of doubles
    std::vector<double> radii;
    std::vector<double> x;
    std::vector<double> xpos0;
    std::vector<double> x0; 
    std::vector<double> up;
    std::vector<double> q;

    std::vector<double> u;
    std::vector<double> udiff;

    std::vector<double> x_n;
    std::vector<double> v_det_n;
    std::vector<double> v_det_c;
    std::vector<double> v_brown;

    std::vector<double> fext; 
    std::vector<double> uext;
    
    std::vector<double> uinf;
    std::vector<double> einf;
    std::vector<double> sh;

    std::vector<double> fb;
    std::vector<double> sb;

    std::vector<double> addrn;
    std::vector<double> drift;

    std::vector<std::vector<long double>> pd; 
    std::vector<std::vector<int>> bond_ids;
    std::vector<std::vector<int>> bond_list; 
    std::vector<int> chain_ids;
    std::vector<std::vector<int>> id;
    std::vector<std::vector<int>> vlist;
    std::vector<int> pair_types;
    std::vector<std::vector<int>> id_AA; 
    std::vector<int> id_AB;
    std::vector<int> id_BB;
    std::vector<std::vector<int>> mesid;

    rank2_array uold;

    rank2_array rfu;
    rank2_array zmuf; 
    rank2_array rfe;
    rank2_array zmus; 
    rank2_array rsu;
    rank2_array rse;
    rank2_array zmes;
    // rank2_array pmob; 

    // rank2_array rfu11;
    // rank2_array rfu22;
    // rank2_array rfu12;
    // rank2_array rfu21;

    // Mat ZMUF; 
    // Mat ZMUS;
    // Mat ZMES;
    // Mat Pinv;
    Mat M;
    Mat Mcm_block;
    Mat Mcc_block;
    Mat Mcc_base;
    Mat B;
    Mat A;

    Vec X;
    Vec Xd;
    Vec Xdet;
    Vec rhs;
    Vec W;

    // Array for temporarily storing the contents of rfu without rotation terms
    rank2_array rtfu;

    // For velocity calculation
    rank2_array rfu_s; 

    // rank2_array zm; // grand mobility matrix

    void initialize_radii(int size){
        radii.resize(size, 0.0);
    }

    void initialize_x(int size){
        x.resize(size, 0.0);
    };

    void initialize_x0(int size){
        x0.resize(size, 0.0);
    };
    
    void initialize_xpos0(int size){
        xpos0.resize(size, 0.0);
    };

    void initialize_up(int size){
        up.resize(size, 0.0);
    };

    void initialize_q(int size){
        q.resize(size, 0.0);
    };

    void initialize_u(int size){
        u.resize(size, 0.0);
    };

    void initialize_udiff(int size){
        udiff.resize(size, 0.0);
    };

    void initialize_x_n(int size){
        x_n.resize(size, 0.0);
    };

    void initialize_v_det_n(int size){
        v_det_n.resize(size, 0.0);
    };

    void initialize_v_det_c(int size){
        v_det_c.resize(size, 0.0);
    };

    void initialize_v_brown(int size){
        v_brown.resize(size, 0.0);
    };

    void initialize_uinf(int size){
        uinf.resize(size, 0.0);
    };

    void initialize_einf(int size){
        einf.resize(size, 0.0);
    };

    void initialize_sh(int size){
        sh.resize(size, 0.0);
    };

    void initialize_fext(int size){
        fext.resize(size, 0.0);
    }

    void initialize_uext(int size){
        uext.resize(size, 0.0);
    }

    void initialize_fb(int size){
        fb.resize(size, 0.0);
    }

    void initialize_sb(int size){
        sb.resize(size, 0.0);
    }

    void initialize_addrn(int size){
        addrn.resize(size, 0.0);
    }

    void initialize_drift(int size){
        drift.resize(size, 0.0);
    }

    void initialize_pd(int rows, int cols){
        pd.resize(rows, std::vector<long double>(cols, 0));
    };

    void initialize_bondid(int rows, int cols){
        bond_ids.resize(rows, std::vector<int>(cols, 0));
    }

    void initialize_chainid(int size){
        chain_ids.resize(size, 0.0);
    }

    void initialize_id(int rows, int cols){
        id.resize(rows, std::vector<int>(cols, 0));
    };

    // void initialize_vlist(int rows, int cols){
    //     vlist.resize(rows, std::vector<int>(cols, 0));
    // };

    void initialize_vlist(int rows){
        vlist.resize(rows);
    };

    void initialize_bond_list(int rows, int cols){
        bond_list.resize(rows, std::vector<int>(cols, 0));
    }

    void initialize_pair_types(int size){
        pair_types.resize(size, 0);
    }
    
    void initialize_id_AA(int rows, int cols){
        id_AA.resize(rows, std::vector<int>(cols, 0));
    }

    void initialize_id_AB(int size){
        id_AB.resize(size, 0);
    }

    void initialize_id_BB(int size){
        id_BB.resize(size, 0);
    }

    void initialize_mesid(int rows, int cols){
        mesid.resize(rows, std::vector<int>(cols,0));
    }

    void initialize_uold(){
        initialize_rank2_array(uold);
    }

    void initialize_rfu(){
        initialize_rank2_array(rfu);
    }

    void initialize_zmuf(){
        initialize_rank2_array(zmuf);
    }

    void initialize_rfe(){
        initialize_rank2_array(rfe);
    }

    void initialize_zmus(){
        initialize_rank2_array(zmus);
    }

    void initialize_rsu(){
        initialize_rank2_array(rsu);
    }

    void initialize_rse(){
        initialize_rank2_array(rse);
    }

    void initialize_zmes(){
        initialize_rank2_array(zmes);
    }

    // void initialize_zm(){
    //     initialize_rank2_array(zm);
    // }

    // void initialize_ZMUF(){
    //     MatCreate(PETSC_COMM_SELF, &ZMUF);
    //     MatSetSizes(ZMUF, PETSC_DECIDE, PETSC_DECIDE, n6p, n6p);
    //     MatSetType(ZMUF, MATDENSE);
    //     MatSetUp(ZMUF);

    // }

    // void initialize_ZMUS(){
    //     MatCreate(PETSC_COMM_SELF, &ZMUS);
    //     MatSetSizes(ZMUS, PETSC_DECIDE, PETSC_DECIDE, n6p, n5p);
    //     MatSetType(ZMUS, MATDENSE);
    //     MatSetUp(ZMUS);
    // }

    // void initialize_ZMES(){
    //     MatCreate(PETSC_COMM_SELF, &ZMES);
    //     MatSetSizes(ZMES, PETSC_DECIDE, PETSC_DECIDE, n5p, n5p);
    //     MatSetType(ZMES, MATDENSE);
    //     MatSetUp(ZMES);
    // }

    void initialize_Pinv(PetscInt nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6){
        PetscErrorCode ierr;
        PetscInt i, ishift, nentries[nm6nc17];

        for( i = 0; i < nm3nc11 ; i++){
            nentries[i] = 1;
        }

        for( i = nm3nc11; i < nm6nc17; i++){
            nentries[i] = 2; 
        }

        ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nm6nc17, nm6nc17, 0, nentries, &Pinv); CHKERRV(ierr);
        for( i = 0; i < nm3nc6; i++){
            ishift = i + nm3nc11;
            ierr = MatSetValue(Pinv, i, ishift, -1.0, INSERT_VALUES); CHKERRV(ierr);
            ierr = MatSetValue(Pinv, ishift, i, -1.0, INSERT_VALUES); CHKERRV(ierr);
            ierr = MatSetValue(Pinv, ishift, ishift, -1.0, INSERT_VALUES); CHKERRV(ierr);
        }

        for( i = nm3nc6; i < nm3nc11; i++){
            ierr = MatSetValue(Pinv, i, i, 1.0, INSERT_VALUES); CHKERRV(ierr);
        }

        ierr = MatAssemblyBegin(Pinv, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);
        ierr = MatAssemblyEnd(Pinv, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);

        // MatView(Pinv, PETSC_VIEWER_STDOUT_WORLD);

    }

    void initialize_M(PetscInt nm3nc11){
        MatCreate(PETSC_COMM_SELF, &M);
        MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, nm3nc11, nm3nc11);
        MatSetType(M, MATDENSE);
        MatSetUp(M);
    }

    void initialize_arrowhead(PetscInt nc11, PetscInt nm3){
        // M^cm : nc11 x nm3 colloid<->monomer coupling (rebuilt each step in mob()).
        MatCreate(PETSC_COMM_SELF, &Mcm_block);
        MatSetSizes(Mcm_block, PETSC_DECIDE, PETSC_DECIDE, nc11, nm3);
        MatSetType(Mcm_block, MATDENSE);
        MatSetUp(Mcm_block);
        // M^cc : nc11 x nc11 colloid self-mobility. Mcc_base holds the raw physics
        // (filled once in fill_self); Mcc_block is the working copy the saddle operator
        // and the Brownian sqrt read, reset from Mcc_base + an adaptive eigen-correction
        // each step (build_slip_vel_schur).
        MatCreate(PETSC_COMM_SELF, &Mcc_base);
        MatSetSizes(Mcc_base, PETSC_DECIDE, PETSC_DECIDE, nc11, nc11);
        MatSetType(Mcc_base, MATDENSE);
        MatSetUp(Mcc_base);
        MatZeroEntries(Mcc_base);

        MatCreate(PETSC_COMM_SELF, &Mcc_block);
        MatSetSizes(Mcc_block, PETSC_DECIDE, PETSC_DECIDE, nc11, nc11);
        MatSetType(Mcc_block, MATDENSE);
        MatSetUp(Mcc_block);
        MatZeroEntries(Mcc_block);
    }

    void initialize_B(PetscInt nm3nc11, PetscInt nm3nc6){   
        PetscErrorCode ierr;
        PetscInt nentries[nm3nc11];
        for( PetscInt i = 0; i < nm3nc6; i++){
            nentries[i] = 1;
        }
        for( PetscInt i = nm3nc6; i < nm3nc11; i++){
            nentries[i] = 0;
        }

        std::cout << "nm3nc11: " << nm3nc11 << std::endl;
        std::cout << "nm3nc6: " << nm3nc6 << std::endl;

        ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nm3nc11,nm3nc6,0,nentries,&B); CHKERRV(ierr);
        for( PetscInt i = 0; i < nm3nc6; i++){
            MatSetValue(B,i,i,-1.0,INSERT_VALUES);
        }

        ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);
        ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);
    }

    void initialize_A(PetscInt nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6){
        
        PetscErrorCode ierr;
        PetscInt i, ishift, Istart, Iend;
        // PetscScalar array4[nm3nc11];
        // PetscScalar arrays5T[nm3nc11*nm3nc6];

        // Dynamic heap allocation
        std::vector<PetscScalar> array4(nm3nc11);
        std::vector<PetscScalar> arrays5T(nm3nc11 * nm3nc6);

        PetscInt rowsUF[nm3nc11], colsUF[nm3nc6], colsUFshift[nm3nc6];
        std::iota(rowsUF, rowsUF + nm3nc11, 0);
        std::iota(colsUF, colsUF + nm3nc6, 0);
        std::iota(colsUFshift, colsUFshift + nm3nc6, nm3nc11);

        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, nm6nc17, nm6nc17, PETSC_DECIDE, PETSC_DECIDE);
        MatSetType(A, MATDENSE); 
        MatSetUp(A);
        
        // ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRV(ierr); 

        // Preallocate with zeros and fill with B and B^T
        ierr = MatZeroEntries(A); CHKERRV(ierr);

        // Insert B into A
        ierr = MatGetValues(B, nm3nc11, rowsUF, nm3nc6, colsUF, arrays5T.data()); CHKERRV(ierr);
        ierr = MatSetValues(A, nm3nc11, rowsUF, nm3nc6, colsUFshift, arrays5T.data(), INSERT_VALUES); CHKERRV(ierr);

        // Insert B^T into A
        for( i = 0; i < nm3nc6; i++){
            ishift = i + nm3nc11;
            MatGetValues(B, nm3nc11, rowsUF, 1, &i, array4.data());
            MatSetValues(A, 1, &ishift, nm3nc11, rowsUF, array4.data(), INSERT_VALUES);
        }

    }

    void initialize_rhs(PetscInt nm6nc17){
        VecCreate(PETSC_COMM_WORLD, &rhs);
        VecSetSizes(rhs, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(rhs);

        // Preallocate with zeros
        // VecZeroEntries(rhs);
    }

    void initialize_W(PetscInt nm3nc11){
        VecCreate(PETSC_COMM_WORLD, &W);
        VecSetSizes(W, PETSC_DECIDE, nm3nc11); 
        VecSetFromOptions(W);
        
    }
    void initialize_X(PetscInt nm6nc17){
        PetscErrorCode ierr;
        VecCreate(PETSC_COMM_WORLD, &X);
        VecSetSizes(X, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(X);
        VecSet(X, 0.0);
        ierr = VecAssemblyBegin(X); CHKERRV(ierr);
        ierr = VecAssemblyEnd(X); CHKERRV(ierr);
    }

    void initialize_Xd(PetscInt nm6nc17){
        PetscErrorCode ierr;
        VecCreate(PETSC_COMM_WORLD, &Xd);
        VecSetSizes(Xd, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(Xd);
        VecSet(Xd, 0.0);
        ierr = VecAssemblyBegin(Xd); CHKERRV(ierr);
        ierr = VecAssemblyEnd(Xd); CHKERRV(ierr);
    }

    void initialize_Xdet(PetscInt nm6nc17){
        PetscErrorCode ierr;
        VecCreate(PETSC_COMM_WORLD, &Xdet);
        VecSetSizes(Xdet, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(Xdet);
        VecSet(Xdet, 0.0);
        ierr = VecAssemblyBegin(Xdet); CHKERRV(ierr);
        ierr = VecAssemblyEnd(Xdet); CHKERRV(ierr);
    }
};