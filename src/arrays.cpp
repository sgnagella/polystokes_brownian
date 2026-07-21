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

    // Cached Schur eigendecomposition (dense path only), refreshed once per stage by
    // sync_mcc_schur_correction() so build_slip_vel_schur() can reuse it without a
    // second syev call. See arrays.h for layout.
    std::vector<PetscScalar> schur_Q;
    std::vector<PetscReal>   schur_lambda_corrected;

    std::vector<double> addrn;
    std::vector<double> drift;

    std::vector<std::vector<long double>> pd; 
    std::vector<std::vector<int>> bond_ids;
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
    Mat Pinv;   // defined (not commented) so arrays.h's `extern Mat Pinv` and the
                // (currently unused) initialize_Pinv() below resolve on a CLEAN build --
                // the old build/ only linked because of a stale pre-comment object file.
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

    MPI_Comm rep_comm = PETSC_COMM_WORLD;   // overridden in init() when mpi_size > 1

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

    // Allocate the (constant) sparsity of the SPD block-diagonal preconditioner
    //     Pinv = diag( D_M^-1 , +D_UF )
    // (the MINRES-compatible reduction of the LDU approximate inverse: the indefinite
    // off-diagonal-coupled form is dropped because MINRES requires an SPD preconditioner).
    // It is purely diagonal -- one nonzero per row. Values are written by fill_Pinv once
    // fill_self() has populated Mcc_base.
    void initialize_Pinv(PetscInt nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6){
        PetscErrorCode ierr;
        std::vector<PetscInt> nentries(nm6nc17, 1);
        ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nm6nc17, nm6nc17, 0, nentries.data(), &Pinv); CHKERRV(ierr);
    }

    // Fill Pinv = diag(D_M^-1, +D_UF). D_M is the mobility self-diagonal (length nm3nc11):
    // beta_inv on the nm3 monomer force DOFs, diag(Mcc_base) on the nc11 colloid FTS DOFs.
    // D_UF is the translation-rotation self-mobility = the first nm3nc6 entries of D_M; it
    // scales the constraint (velocity) block, replacing Jacobi's placeholder 1 with the true
    // velocity-mobility scale. The force/FTS block gets D_M^-1 (same as Jacobi there).
    void fill_Pinv(PetscInt nm3, PetscInt nc11, PetscInt nm3nc6, PetscInt nm3nc11,
                   PetscReal beta_inv){
        PetscErrorCode ierr;

        // D_M (length nm3nc11): beta_inv on monomers, diag(Mcc_base) on the colloid.
        std::vector<PetscScalar> DM(nm3nc11, 0.0);
        for( PetscInt i = 0; i < nm3; i++) DM[i] = beta_inv;
        {
            Vec dc;
            ierr = VecCreateSeq(PETSC_COMM_SELF, nc11, &dc); CHKERRV(ierr);
            ierr = MatGetDiagonal(Mcc_base, dc); CHKERRV(ierr);
            const PetscScalar *d;
            ierr = VecGetArrayRead(dc, &d); CHKERRV(ierr);
            for( PetscInt k = 0; k < nc11; k++) DM[nm3 + k] = d[k];
            ierr = VecRestoreArrayRead(dc, &d); CHKERRV(ierr);
            ierr = VecDestroy(&dc); CHKERRV(ierr);
        }

        // Force/FTS block: D_M^-1.
        for( PetscInt i = 0; i < nm3nc11; i++){
            ierr = MatSetValue(Pinv, i, i, 1.0/DM[i], INSERT_VALUES); CHKERRV(ierr);
        }
        // Constraint (velocity) block: +D_UF_j = DM[j] for j < nm3nc6.
        for( PetscInt j = 0; j < nm3nc6; j++){
            ierr = MatSetValue(Pinv, j + nm3nc11, j + nm3nc11, DM[j], INSERT_VALUES); CHKERRV(ierr);
        }

        ierr = MatAssemblyBegin(Pinv, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);
        ierr = MatAssemblyEnd(Pinv, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);
        // MatView(Pinv, PETSC_VIEWER_STDOUT_WORLD);
    }

    // Generic PCSHELL apply: y = P x, where P is the Mat set as the shell context. Serial uses
    // arrays::Pinv; the MPI path uses its own rank-local Pinv (matfree_A_mpi.cpp).
    PetscErrorCode PinvShellApply(PC pc, Vec x, Vec y){
        Mat P;
        PetscErrorCode ierr = PCShellGetContext(pc, (void**)&P); CHKERRQ(ierr);
        ierr = MatMult(P, x, y); CHKERRQ(ierr);
        return 0;
    }

    void initialize_M(PetscInt nm3nc11){
        MatCreate(PETSC_COMM_SELF, &M);
        MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, nm3nc11, nm3nc11);
        MatSetType(M, MATDENSE);
        MatSetUp(M);
    }

    void initialize_arrowhead(PetscInt nc11, PetscInt mcm_cols){
        // M^cm : nc11 x mcm_cols colloid<->monomer coupling (rebuilt each step in mob()).
        // mcm_cols = nm3 in serial, or this rank's local 3*nloc under MPI (only local monomer
        // columns are stored/filled; the Schur and slip-vel contractions Allreduce the partials).
        MatCreate(PETSC_COMM_SELF, &Mcm_block);
        MatSetSizes(Mcm_block, PETSC_DECIDE, PETSC_DECIDE, nc11, mcm_cols);
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
        VecCreate(rep_comm, &rhs);
        VecSetSizes(rhs, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(rhs);

        // Preallocate with zeros
        // VecZeroEntries(rhs);
    }

    void initialize_W(PetscInt nm3nc11){
        VecCreate(rep_comm, &W);
        VecSetSizes(W, PETSC_DECIDE, nm3nc11); 
        VecSetFromOptions(W);
        
    }
    void initialize_X(PetscInt nm6nc17){
        PetscErrorCode ierr;
        VecCreate(rep_comm, &X);
        VecSetSizes(X, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(X);
        VecSet(X, 0.0);
        ierr = VecAssemblyBegin(X); CHKERRV(ierr);
        ierr = VecAssemblyEnd(X); CHKERRV(ierr);
    }

    void initialize_Xd(PetscInt nm6nc17){
        PetscErrorCode ierr;
        VecCreate(rep_comm, &Xd);
        VecSetSizes(Xd, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(Xd);
        VecSet(Xd, 0.0);
        ierr = VecAssemblyBegin(Xd); CHKERRV(ierr);
        ierr = VecAssemblyEnd(Xd); CHKERRV(ierr);
    }

    void initialize_Xdet(PetscInt nm6nc17){
        PetscErrorCode ierr;
        VecCreate(rep_comm, &Xdet);
        VecSetSizes(Xdet, PETSC_DECIDE, nm6nc17);
        VecSetFromOptions(Xdet);
        VecSet(Xdet, 0.0);
        ierr = VecAssemblyBegin(Xdet); CHKERRV(ierr);
        ierr = VecAssemblyEnd(Xdet); CHKERRV(ierr);
    }
};